# -*- coding: utf-8 -*-
"""
DMN-ZS Global Runner - 鑱旈偊 DMN 闆舵牱鏈祴璇曟椂閫傚簲
涓ユ牸澶嶇幇 DMN-ZS 鍘熻鏂囷紝鏀寔 local / global(share_cache) 涓ょ妯″紡
鏁版嵁鍔犺浇鍜屽鎴风鍒嗛厤閫昏緫涓?tda_runner.py 淇濇寔涓€鑷?"""

import math
import random
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import clip
import yaml

from utils import *
from datasets.federated_loader import FederatedDataset, cache_features


# ==================== 鍙傛暟瑙ｆ瀽 ====================

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--wandb-log', dest='wandb', action='store_true')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True)
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True)
    parser.add_argument('--num-clients', dest='num_clients', type=int, default=10)
    parser.add_argument('--part-rate', dest='part_rate', type=float, default=1.0)
    parser.add_argument('--sync-freq', dest='sync_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cache-features', dest='cache_features', action='store_true')
    parser.add_argument('--separate-domains', dest='separate_domains', action='store_true')
    args = parser.parse_args()
    return args


# ==================== DMN 鏍稿績锛氬唴瀛樺簱 ====================

class DMNMemory:
    """
    DMN-ZS 闆舵牱鏈増鏈殑鍙屽唴瀛樺簱銆?    鍙娇鐢?image memory锛堝姩鎬侊級锛宼ext feat 浣滀负鍥哄畾鍏堥獙銆?    涓嶅寘鍚彲瀛︿範鍙傛暟锛圸S 鐗堟湰锛夈€?    """
    def __init__(self, num_class, feat_dim, memory_size, device, dtype):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.memory_size = memory_size
        self.device = device
        self.dtype = dtype

        # image memory: [num_class, memory_size, feat_dim]
        self.image_feature_memory = torch.zeros(num_class, memory_size, feat_dim,
                                                device=device, dtype=dtype)
        # entropy memory: [num_class, memory_size]
        self.image_entropy_mem = torch.zeros(num_class, memory_size,
                                             device=device, dtype=dtype)
        # count: [num_class, 1]
        self.image_feature_count = torch.zeros(num_class, 1, device=device, dtype=torch.long)

    def update(self, image_feat, pseudo_label, entropy):
        """
        鏇存柊 image memory锛屼繚鐣欎綆鐔垫牱鏈€?        - image_feat: [1, feat_dim]
        - pseudo_label: int
        - entropy: float
        """
        c = pseudo_label
        feat = image_feat.squeeze(0)  # [feat_dim]
        if self.image_feature_count[c, 0].item() == self.memory_size:
            # 婊′簡锛氬彧鏈夊綋鍓嶆牱鏈喌鏇翠綆鏃舵墠鏇挎崲鏈€楂樼喌鐨?            if (entropy < self.image_entropy_mem[c]).any():
                max_idx = self.image_entropy_mem[c].argmax().item()
                self.image_feature_memory[c, max_idx] = feat
                self.image_entropy_mem[c, max_idx] = entropy
        else:
            idx = self.image_feature_count[c, 0].item()
            self.image_feature_memory[c, idx] = feat
            self.image_entropy_mem[c, idx] = entropy
            self.image_feature_count[c] += 1


# ==================== BaseClient ====================

class BaseClient:
    def __init__(self, dataset, clip_weights, args):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.curr_idx = 0
        self.clip_weights = clip_weights          # [num_class, feat_dim]
        self.num_class, self.feat_dim = clip_weights.shape
        self.dtype = clip_weights.dtype
        self.device = clip_weights.device
        self.num_correct = 0
        self.args = args
        self.clip_model = getattr(args, 'clip_model', None)

    @torch.no_grad()
    def predict(self, image_feature):
        clip_logits, pred, _, _ = get_clip_logits(image_feature, self.clip_weights, normalize=False)
        return pred

    def evaluate_one(self):
        if self.curr_idx >= self.num_samples:
            return 0, 0
        data, label = self.dataset[self.curr_idx]
        self.curr_idx += 1

        dataset = self.dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        is_cached = hasattr(dataset, 'is_cached_features') and dataset.is_cached_features

        if data.dim() == 3 or (data.dim() == 4 and data.shape[0] == 1):
            if data.dim() == 3:
                data = data.unsqueeze(0)
            data = data.to(device=self.device)
            with torch.no_grad():
                image_feature = self.clip_model.encode_image(data)
                image_feature = image_feature.to(dtype=self.dtype)
                image_feature = F.normalize(image_feature, dim=1)
        else:
            image_feature = data.to(device=self.device, dtype=self.dtype)
            if image_feature.dim() == 1:
                image_feature = image_feature.unsqueeze(0)
            if not is_cached:
                image_feature = F.normalize(image_feature, dim=1)

        pred = self.predict(image_feature)
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)
        correct = int(pred == label_int)
        self.num_correct += correct
        return correct, 1

    def is_done(self):
        return self.curr_idx >= self.num_samples


# ==================== DMNClient ====================

class DMNClient(BaseClient):
    """
    DMN-ZS 瀹㈡埛绔細
    - 闆舵牱鏈紝涓嶉渶瑕佸彲瀛︿範鍙傛暟
    - image memory 淇濆瓨浣庣喌鏍锋湰鐗瑰緛
    - text feat 浣滀负鍥哄畾鍏堥獙锛堝浐瀹氬叏灞€鐗瑰緛锛?    - 鐢ㄧ浉浼煎害鍔犳潈鐨勬柟寮忚瀺鍚?memory 寰楀埌鏈€缁堥娴?    """

    def __init__(self, dataset, clip_weights, args):
        super(DMNClient, self).__init__(dataset, clip_weights, args)

        cfg = args.config.get('dmn', {})
        self.memory_size = cfg.get('memory_size', 50)
        self.beta = cfg.get('beta', 10.0)
        # beta2锛氬師璁烘枃浜嬪悗缃戞牸鎼滅储鏈€浼樿瀺鍚堟潈閲嶏紙pred_vanilla + beta2 * pred_global锛?        # 鍦ㄧ嚎鍦烘櫙鏃犳硶浜嬪悗鎼滅储锛屼粠閰嶇疆璇诲彇锛岄粯璁?0.5
        self.beta2 = cfg.get('beta2', 0.5)

        # logit_scale 浠?clip 妯″瀷鍔ㄦ€佽幏鍙栵紝涓庡師璁烘枃涓€鑷?        clip_model = getattr(args, 'clip_model', None)
        if clip_model is not None and hasattr(clip_model, 'logit_scale'):
            self.logit_scale = clip_model.logit_scale.exp().item()
        else:
            self.logit_scale = 100.0

        # text feat 浣滀负鍥哄畾鍏堥獙锛歔num_class, 1, feat_dim]
        # clip_weights: [num_class, feat_dim]
        self.fixed_global_feat = clip_weights.unsqueeze(1).clone()  # [num_class, 1, feat_dim]

        # image memory
        self.memory = DMNMemory(
            num_class=self.num_class,
            feat_dim=self.feat_dim,
            memory_size=self.memory_size,
            device=self.device,
            dtype=self.dtype,
        )

    def _softmax_entropy(self, prob):
        """Compute entropy for probability vectors."""
        return -(prob * torch.log(prob + 1e-8)).sum(dim=-1)

    def _get_text_pred(self, image_feat):
        """Return zero-shot CLIP prediction probabilities."""
        logits = self.logit_scale * image_feat @ self.clip_weights.T
        return logits.softmax(dim=-1)

    def _get_image_memory_pred(self, image_feat):
        """Predict with image memory plus text prior."""
        combined = torch.cat([self.memory.image_feature_memory,
                              self.fixed_global_feat], dim=1)

        empty_mask = combined.norm(dim=-1) < 1e-6
        combined_norm = combined.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        combined_K = combined / combined_norm
        combined_V = combined / combined_norm
        combined_K[empty_mask] = 0.0
        combined_V[empty_mask] = 0.0

        similarity = (image_feat.unsqueeze(0) * combined_K).sum(-1)
        similarity = torch.exp(-self.beta * (1.0 - similarity))
        similarity[empty_mask] = 0.0

        denom = similarity.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weight = similarity / denom
        visual_proto = (weight.unsqueeze(-1) * combined_V).sum(dim=1)
        visual_proto = F.normalize(visual_proto, dim=-1)

        logits = self.logit_scale * image_feat @ visual_proto.T
        return logits.softmax(dim=-1)

# -*- coding: utf-8 -*-
"""
DMN-ZS Global Runner - 鑱旈偊 DMN 闆舵牱鏈祴璇曟椂閫傚簲
涓ユ牸澶嶇幇 DMN-ZS 鍘熻鏂囷紝鏀寔 local / global(share_cache) 涓ょ妯″紡
鏁版嵁鍔犺浇鍜屽鎴风鍒嗛厤閫昏緫涓?tda_runner.py 淇濇寔涓€鑷?"""

import math
import random
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import clip
import yaml

from utils import *
from datasets.federated_loader import FederatedDataset, cache_features


# ==================== 鍙傛暟瑙ｆ瀽 ====================

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--wandb-log', dest='wandb', action='store_true')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True)
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True)
    parser.add_argument('--num-clients', dest='num_clients', type=int, default=10)
    parser.add_argument('--part-rate', dest='part_rate', type=float, default=1.0)
    parser.add_argument('--sync-freq', dest='sync_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cache-features', dest='cache_features', action='store_true')
    parser.add_argument('--separate-domains', dest='separate_domains', action='store_true')
    args = parser.parse_args()
    return args


# ==================== DMN 鏍稿績锛氬唴瀛樺簱 ====================

class DMNMemory:
    """
    DMN-ZS 闆舵牱鏈増鏈殑鍙屽唴瀛樺簱銆?    鍙娇鐢?image memory锛堝姩鎬侊級锛宼ext feat 浣滀负鍥哄畾鍏堥獙銆?    涓嶅寘鍚彲瀛︿範鍙傛暟锛圸S 鐗堟湰锛夈€?    """
    def __init__(self, num_class, feat_dim, memory_size, device, dtype):
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.memory_size = memory_size
        self.device = device
        self.dtype = dtype

        # image memory: [num_class, memory_size, feat_dim]
        self.image_feature_memory = torch.zeros(num_class, memory_size, feat_dim,
                                                device=device, dtype=dtype)
        # entropy memory: [num_class, memory_size]
        self.image_entropy_mem = torch.zeros(num_class, memory_size,
                                             device=device, dtype=dtype)
        # count: [num_class, 1]
        self.image_feature_count = torch.zeros(num_class, 1, device=device, dtype=torch.long)

    def update(self, image_feat, pseudo_label, entropy):
        """
        鏇存柊 image memory锛屼繚鐣欎綆鐔垫牱鏈€?        - image_feat: [1, feat_dim]
        - pseudo_label: int
        - entropy: float
        """
        c = pseudo_label
        feat = image_feat.squeeze(0)  # [feat_dim]
        if self.image_feature_count[c, 0].item() == self.memory_size:
            # 婊′簡锛氬彧鏈夊綋鍓嶆牱鏈喌鏇翠綆鏃舵墠鏇挎崲鏈€楂樼喌鐨?            if (entropy < self.image_entropy_mem[c]).any():
                max_idx = self.image_entropy_mem[c].argmax().item()
                self.image_feature_memory[c, max_idx] = feat
                self.image_entropy_mem[c, max_idx] = entropy
        else:
            idx = self.image_feature_count[c, 0].item()
            self.image_feature_memory[c, idx] = feat
            self.image_entropy_mem[c, idx] = entropy
            self.image_feature_count[c] += 1


# ==================== BaseClient ====================

class BaseClient:
    def __init__(self, dataset, clip_weights, args):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.curr_idx = 0
        self.clip_weights = clip_weights          # [num_class, feat_dim]
        self.num_class, self.feat_dim = clip_weights.shape
        self.dtype = clip_weights.dtype
        self.device = clip_weights.device
        self.num_correct = 0
        self.args = args
        self.clip_model = getattr(args, 'clip_model', None)

    @torch.no_grad()
    def predict(self, image_feature):
        clip_logits, pred, _, _ = get_clip_logits(image_feature, self.clip_weights, normalize=False)
        return pred

    def evaluate_one(self):
        if self.curr_idx >= self.num_samples:
            return 0, 0
        data, label = self.dataset[self.curr_idx]
        self.curr_idx += 1

        dataset = self.dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        is_cached = hasattr(dataset, 'is_cached_features') and dataset.is_cached_features

        if data.dim() == 3 or (data.dim() == 4 and data.shape[0] == 1):
            if data.dim() == 3:
                data = data.unsqueeze(0)
            data = data.to(device=self.device)
            with torch.no_grad():
                image_feature = self.clip_model.encode_image(data)
                image_feature = image_feature.to(dtype=self.dtype)
                image_feature = F.normalize(image_feature, dim=1)
        else:
            image_feature = data.to(device=self.device, dtype=self.dtype)
            if image_feature.dim() == 1:
                image_feature = image_feature.unsqueeze(0)
            if not is_cached:
                image_feature = F.normalize(image_feature, dim=1)

        pred = self.predict(image_feature)
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)
        correct = int(pred == label_int)
        self.num_correct += correct
        return correct, 1

    def is_done(self):
        return self.curr_idx >= self.num_samples


# ==================== DMNClient ====================

class DMNClient(BaseClient):
    """
    DMN-ZS 瀹㈡埛绔細
    - 闆舵牱鏈紝涓嶉渶瑕佸彲瀛︿範鍙傛暟
    - image memory 淇濆瓨浣庣喌鏍锋湰鐗瑰緛
    - text feat 浣滀负鍥哄畾鍏堥獙锛堝浐瀹氬叏灞€鐗瑰緛锛?    - 鐢ㄧ浉浼煎害鍔犳潈鐨勬柟寮忚瀺鍚?memory 寰楀埌鏈€缁堥娴?    """

    def __init__(self, dataset, clip_weights, args):
        super(DMNClient, self).__init__(dataset, clip_weights, args)

        cfg = args.config.get('dmn', {})
        self.memory_size = cfg.get('memory_size', 50)
        self.beta = cfg.get('beta', 10.0)
        # beta2锛氬師璁烘枃浜嬪悗缃戞牸鎼滅储鏈€浼樿瀺鍚堟潈閲嶏紙pred_vanilla + beta2 * pred_global锛?        # 鍦ㄧ嚎鍦烘櫙鏃犳硶浜嬪悗鎼滅储锛屼粠閰嶇疆璇诲彇锛岄粯璁?0.5
        self.beta2 = cfg.get('beta2', 0.5)

        # logit_scale 浠?clip 妯″瀷鍔ㄦ€佽幏鍙栵紝涓庡師璁烘枃涓€鑷?        clip_model = getattr(args, 'clip_model', None)
        if clip_model is not None and hasattr(clip_model, 'logit_scale'):
            self.logit_scale = clip_model.logit_scale.exp().item()
        else:
            self.logit_scale = 100.0

        # text feat 浣滀负鍥哄畾鍏堥獙锛歔num_class, 1, feat_dim]
        # clip_weights: [num_class, feat_dim]
        self.fixed_global_feat = clip_weights.unsqueeze(1).clone()  # [num_class, 1, feat_dim]

        # image memory
        self.memory = DMNMemory(
            num_class=self.num_class,
            feat_dim=self.feat_dim,
            memory_size=self.memory_size,
            device=self.device,
            dtype=self.dtype,
        )

    def _softmax_entropy(self, prob):
        """Compute entropy for probability vectors."""
        return -(prob * torch.log(prob + 1e-8)).sum(dim=-1)

    def _get_text_pred(self, image_feat):
        """Return zero-shot CLIP prediction probabilities."""
        logits = self.logit_scale * image_feat @ self.clip_weights.T
        return logits.softmax(dim=-1)

    def _get_image_memory_pred(self, image_feat):
        """Predict with image memory plus text prior."""
        combined = torch.cat([self.memory.image_feature_memory,
                              self.fixed_global_feat], dim=1)

        empty_mask = combined.norm(dim=-1) < 1e-6
        combined_norm = combined.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        combined_K = combined / combined_norm
        combined_V = combined / combined_norm
        combined_K[empty_mask] = 0.0
        combined_V[empty_mask] = 0.0

        similarity = (image_feat.unsqueeze(0) * combined_K).sum(-1)
        similarity = torch.exp(-self.beta * (1.0 - similarity))

        adaptive_feat = (combined_V * similarity.unsqueeze(-1)).sum(1)
        norm = adaptive_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        adaptive_feat = adaptive_feat / norm

        logits = self.logit_scale * adaptive_feat @ image_feat.T
        return logits.T.softmax(dim=-1)

    @torch.no_grad()
    def predict(self, image_feature):
        text_prob = self._get_text_pred(image_feature)
        pseudo_label = text_prob.argmax(dim=1).item()
        entropy = self._softmax_entropy(text_prob).item()
        self.memory.update(image_feature, pseudo_label, entropy)

        img_prob = self._get_image_memory_pred(image_feature)
        final_prob = text_prob + self.beta2 * img_prob
        return final_prob.argmax(dim=1).item()
# ==================== BaseCTTAServer ====================

class BaseCTTAServer:
    def __init__(self, datasets, clip_weights, args, client_class=BaseClient):
        self.num_class, self.feat_dim = clip_weights.shape
        self.clients = OrderedDict(
            [(cid, client_class(dataset, clip_weights, args))
             for (cid, dataset) in datasets.items()])
        self.client_ids = list(datasets.keys())
        self.num_clients = len(datasets)
        self.dtype = clip_weights.dtype
        self.device = clip_weights.device
        self.cohort_size = int(self.num_clients * args.part_rate)
        self.num_rounds = sum(client.num_samples for client in self.clients.values())
        self.sync_freq = args.sync_freq

    def evaluate(self):
        total_correct, total_num_samples = 0, 0
        for rnd in tqdm(range(1, self.num_rounds + 1), desc='Federated TTA'):
            selected_idx = sorted(list(torch.randperm(self.num_clients)[:self.cohort_size].numpy()))
            rnd_correct, rnd_num_samples = 0, 0
            for idx in selected_idx:
                client = self.clients[self.client_ids[idx]]
                correct, num_samples = client.evaluate_one()
                rnd_correct += correct
                rnd_num_samples += num_samples
            total_correct += rnd_correct
            total_num_samples += rnd_num_samples
            if all(client.is_done() for client in self.clients.values()):
                break
            if rnd % self.sync_freq == 0:
                self.syncronize()
        stats = {cid: client.num_correct / client.num_samples
                 for (cid, client) in self.clients.items()}
        return total_correct / total_num_samples, total_correct, total_num_samples, stats

    def syncronize(self):
        pass


# ==================== DMNServer ====================

class DMNServer(BaseCTTAServer):
    def __init__(self, datasets, clip_weights, args):
        super(DMNServer, self).__init__(datasets, clip_weights, args, client_class=DMNClient)


# ==================== 涓诲嚱鏁?====================

def main():
    args = get_arguments()
    config_path = args.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    result = clip.load(args.backbone, device=device)
    if len(result) == 3:
        clip_model, embed_dim, preprocess = result
    else:
        clip_model, preprocess = result
    clip_model.eval()
    args.clip_model = clip_model
    args.device = device

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.wandb:
        date = datetime.now().strftime('%b%d_%H-%M-%S')
        group_name = f'{args.backbone}_{args.datasets}_{date}'

    # 閰嶇疆鏂囦欢鍚嶇О鏄犲皠
    _dmn_name_map = {
        'VLCS': 'vlcs',
        'TerraIncognita': 'terra',
        'OfficeHome': 'officehome',
        'CIFAR10CFull': 'cifar10c',
        'CIFAR100CFull': 'cifar100c',
    }

    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f'Processing {dataset_name} dataset.')

        # Load DMN-ZS configs. Keep dmn_*.yaml as a backward-compatible fallback.
        _dmn_suffix = _dmn_name_map.get(dataset_name, dataset_name.lower())
        _dmn_config_file = os.path.join(config_path, f'dmn_zs_{_dmn_suffix}.yaml')
        if not os.path.exists(_dmn_config_file):
            _dmn_config_file = os.path.join(config_path, f'dmn_{_dmn_suffix}.yaml')
        if not os.path.exists(_dmn_config_file):
            _dmn_config_file = os.path.join(os.path.dirname(__file__), 'configs', f'dmn_zs_{_dmn_suffix}.yaml')
        if not os.path.exists(_dmn_config_file):
            _dmn_config_file = os.path.join(os.path.dirname(__file__), 'configs', f'dmn_{_dmn_suffix}.yaml')
        if os.path.exists(_dmn_config_file):
            with open(_dmn_config_file, 'r', encoding='utf-8') as _f:
                cfg = yaml.load(_f, Loader=yaml.SafeLoader)
        else:
            cfg = get_config_file(config_path, dataset_name)
        print('\nRunning dataset configurations:')
        print(cfg, '\n')
        args.config = cfg

        dmn_cfg = cfg.get('dmn', {})
        share_cache = dmn_cfg.get('share_cache', False)

        if args.separate_domains:
            # ===== 鍒嗗煙璇勪及妯″紡 =====
            domain_loaders, classnames, template = build_test_data_loader(
                dataset_name, args.data_root, preprocess, separate_domains=True)
            clip_weights = clip_classifier(classnames, template, clip_model)

            print(f"\n{'='*60}")
            print(f'Evaluating {dataset_name} - Separate Domain Mode')
            print(f"{'='*60}\n")

            domain_results = {}
            total_correct_all = 0
            total_samples_all = 0

            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f'\n--- Evaluating domain: {domain_name} ---')
                dataset = domain_loader.dataset

                if args.cache_features:
                    print(f'Pre-caching CLIP features for {domain_name}...')
                    cache_name = f'{dataset_name}_{domain_name}_{args.backbone.replace("/", "_")}'
                    dataset = cache_features(dataset, clip_model, device=device,
                                            cache_dir='./cached_features', dataset_name=cache_name)

                total_samples_domain = len(dataset)
                samples_per_client = total_samples_domain // args.num_clients
                from torch.utils.data import Subset
                client_datasets = OrderedDict()
                rng = np.random.RandomState(args.seed + domain_idx)
                shuffled = rng.permutation(total_samples_domain)
                for i in range(args.num_clients):
                    start_idx = i * samples_per_client
                    end_idx = start_idx + samples_per_client if i < args.num_clients - 1 else total_samples_domain
                    client_datasets[f'client_{i}'] = Subset(dataset, shuffled[start_idx:end_idx].tolist())

                print(f'Created {len(client_datasets)} clients for {domain_name}.')

                server = DMNServer(client_datasets, clip_weights, args)

                # 鍩熷唴鍏变韩鍐呭瓨
                if share_cache:
                    _clients = list(server.clients.values())
                    if len(_clients) > 1:
                        _c0 = _clients[0]
                        for _c in _clients[1:]:
                            _c.memory = _c0.memory
                        print(f'  [share_cache] {len(_clients)} clients share memory (separate-domains mode).')

                if args.wandb:
                    run_name = f'{dataset_name}_{domain_name}'
                    run = wandb.init(project='DMN-CTTA', config=cfg, group=group_name, name=run_name)

                acc, total_correct, total_samples, stats = server.evaluate()
                domain_results[domain_name] = acc
                total_correct_all += total_correct
                total_samples_all += total_samples

                print(f'{domain_name}: {acc * 100:.2f}% ({total_correct}/{total_samples})')

                if args.wandb:
                    wandb.log({f'{domain_name}_acc': acc * 100})
                    run.finish()

            overall_acc = total_correct_all / total_samples_all if total_samples_all > 0 else 0.0
            print(f"\n{'='*60}")
            print(f'Overall Results ({dataset_name}):')
            print(f"{'='*60}")
            for dname, dacc in domain_results.items():
                print(f'  {dname}: {dacc * 100:.2f}%')
            print(f'  Average: {overall_acc * 100:.2f}%')
            print(f"{'='*60}\n")

        else:
            # ===== 鍚堝苟璇勪及妯″紡锛堝叏灞€鍏变韩锛?====
            domain_loaders, classnames, template = build_test_data_loader(
                dataset_name, args.data_root, preprocess, separate_domains=True)
            clip_weights = clip_classifier(classnames, template, clip_model)

            num_domains = len(domain_loaders)
            total_clients = num_domains * args.num_clients

            print(f"\n{'='*60}")
            print(f'Evaluating {dataset_name} - Collaborative Mode')
            print(f'Strategy: {num_domains} domains x {args.num_clients} clients = {total_clients} total')
            print(f'Participation rate: {args.part_rate} ({int(total_clients * args.part_rate)} clients per round)')
            print(f"{'='*60}\n")

            all_client_datasets = OrderedDict()

            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f'Setting up clients for {domain_name}...')
                dataset = domain_loader.dataset

                if args.cache_features:
                    cache_name = f'{dataset_name}_{domain_name}_{args.backbone.replace("/", "_")}'
                    dataset = cache_features(dataset, clip_model, device=device,
                                            cache_dir='./cached_features', dataset_name=cache_name)

                total_samples = len(dataset)
                samples_per_client = total_samples // args.num_clients

                from torch.utils.data import Subset
                rng = np.random.RandomState(args.seed + domain_idx)
                indices_pool = rng.permutation(total_samples).tolist()

                for i in range(args.num_clients):
                    start_idx = i * samples_per_client
                    end_idx = start_idx + samples_per_client if i < args.num_clients - 1 else total_samples
                    all_client_datasets[f'{domain_name}_client_{i}'] = Subset(dataset, indices_pool[start_idx:end_idx])

                print(f'  Created {args.num_clients} clients, each with ~{samples_per_client} samples')

            print(f'\nTotal clients: {len(all_client_datasets)}')

            if args.wandb:
                run_name = f'{dataset_name}_collaborative'
                run = wandb.init(project='DMN-CTTA', config=cfg, group=group_name, name=run_name)

            server = DMNServer(all_client_datasets, clip_weights, args)

            # 鍏ㄥ眬鍏变韩鍐呭瓨
            if share_cache:
                _clients = list(server.clients.values())
                if len(_clients) > 1:
                    _c0 = _clients[0]
                    for _c in _clients[1:]:
                        _c.memory = _c0.memory
                    print(f'  [share_cache] {len(_clients)} clients share memory (global mode).')

            acc, total_correct, total_samples, stats = server.evaluate()

            print(f"\n{'='*60}")
            print(f'Overall Results ({dataset_name}):')
            print(f"{'='*60}")
            print(f'Total Accuracy: {acc * 100:.2f}% ({total_correct}/{total_samples})')

            domain_results = {}
            for domain_name in domain_loaders.keys():
                d_correct = sum(server.clients[f'{domain_name}_client_{i}'].num_correct
                                for i in range(args.num_clients)
                                if f'{domain_name}_client_{i}' in server.clients)
                d_total = sum(server.clients[f'{domain_name}_client_{i}'].num_samples
                              for i in range(args.num_clients)
                              if f'{domain_name}_client_{i}' in server.clients)
                if d_total > 0:
                    domain_results[domain_name] = d_correct / d_total
                    print(f'  {domain_name}: {domain_results[domain_name] * 100:.2f}% ({d_correct}/{d_total})')

            print(f"{'='*60}\n")

            if args.wandb:
                wandb.log({'overall_acc': acc * 100})
                for dname, dacc in domain_results.items():
                    wandb.log({f'{dname}_acc': dacc * 100})
                run.finish()


if __name__ == '__main__':
    main()

