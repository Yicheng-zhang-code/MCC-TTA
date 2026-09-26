# -*- coding: utf-8 -*-
"""
Zero Runner - 鑱旈偊 Zero 娴嬭瘯鏃堕€傚簲鍩虹嚎

澶嶇幇 Zero 鍘熻鏂囨牳蹇冪畻娉曪紙ICLR 2024锛夛細
  - 姣忎釜娴嬭瘯鏍锋湰鐢熸垚 num_views 涓寮鸿鍥?
  - 璁＄畻鎵€鏈夎鍥?image features
  - confidence filter 淇濈暀浣庣喌鍓?gamma 姣斾緥瑙嗗浘
  - zero-temperature softmax + 鎶曠エ棰勬祴

鏁版嵁鍔犺浇銆佸鎴风鍒嗛厤銆佽仈閭︾粨鏋勩€佽緭鍑烘ā寮忎笌 statA(local).py 瀹屽叏涓€鑷淬€?
鏍稿績鏂规硶涓嶅悓锛歓ero 闇€瑕佸師濮嬪浘鐗囧仛澶氳鍥惧寮猴紝涓嶆敮鎸侀缂撳瓨鐗瑰緛妯″紡銆?
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import argparse
import yaml
import wandb
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import clip
from utils import *


# ==================== 鍙傛暟瑙ｆ瀽锛堜笌 statA(local).py 瀹屽叏涓€鑷达級====================

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('--wandb-log', dest='wandb', action='store_true')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True)
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str,
                        choices=['RN50', 'ViT-B/16'], required=True)
    parser.add_argument('--num-clients', dest='num_clients', type=int, default=10)
    parser.add_argument('--part-rate', dest='part_rate', type=float, default=1.0)
    parser.add_argument('--sync-freq', dest='sync_freq', type=int, default=100)
    parser.add_argument('--partition', dest='partition', type=str, default='iid',
                        choices=['iid', 'domain', 'corruption'])
    parser.add_argument('--seed', type=int, default=1)
    # Zero uses raw images for multi-view augmentation; this flag is kept for CLI compatibility.
    parser.add_argument('--cache-features', dest='cache_features', action='store_true',
                        help='Kept for CLI compatibility; Zero does not use cached features.')
    parser.add_argument('--separate-domains', dest='separate_domains', action='store_true')
    parser.add_argument('--max-rounds', dest='max_rounds', type=int, default=-1,
                        help='Maximum evaluation rounds. Use -1 to run the full dataset.')
    args = parser.parse_args()
    return args


# ==================== Zero GPU 鎵归噺 AugMix-like 澶氳鍥惧寮?====================


class AugMixAugmenter:
    def __init__(self, augmix=True):
        self.augmix = augmix

    @torch.no_grad()
    def __call__(self, img_tensor, num_views, device, dtype):
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=device, dtype=torch.float32).view(1, 3, 1, 1)

        orig = img_tensor.to(device=device, dtype=torch.float32).unsqueeze(0)
        if num_views <= 1:
            return orig.to(dtype=dtype)

        n_aug = num_views - 1
        x = (orig * std + mean).clamp(0.0, 1.0)
        x = x.expand(n_aug, -1, -1, -1).contiguous()
        n, _, h, w = x.shape

        scale = 0.5 + 0.5 * torch.rand(n, device=device)
        log_ratio_min = torch.log(torch.tensor(3.0 / 4.0, device=device))
        log_ratio_max = torch.log(torch.tensor(4.0 / 3.0, device=device))
        ratio = torch.exp(log_ratio_min + (log_ratio_max - log_ratio_min) * torch.rand(n, device=device))

        crop_h = (torch.sqrt(scale / ratio) * h).clamp(1.0, float(h))
        crop_w = (torch.sqrt(scale * ratio) * w).clamp(1.0, float(w))

        cx_min = (crop_w / 2.0) / w * 2.0 - 1.0
        cx_max = 1.0 - (crop_w / 2.0) / w * 2.0
        cy_min = (crop_h / 2.0) / h * 2.0 - 1.0
        cy_max = 1.0 - (crop_h / 2.0) / h * 2.0

        tx = cx_min + (cx_max - cx_min) * torch.rand(n, device=device)
        ty = cy_min + (cy_max - cy_min) * torch.rand(n, device=device)

        theta = torch.zeros(n, 2, 3, device=device, dtype=torch.float32)
        theta[:, 0, 0] = crop_w / w
        theta[:, 1, 1] = crop_h / h
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, size=(n, 3, 224, 224), align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

        flip_mask = torch.rand(n, device=device) > 0.5
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])

        if self.augmix:
            brightness = 0.6 + 0.8 * torch.rand(n, device=device)
            x = x * brightness.view(n, 1, 1, 1)

            contrast = 0.6 + 0.8 * torch.rand(n, device=device)
            x_mean = x.mean(dim=(2, 3), keepdim=True)
            x = (x - x_mean) * contrast.view(n, 1, 1, 1) + x_mean

            saturation = 0.6 + 0.8 * torch.rand(n, device=device)
            gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
            x = (x - gray) * saturation.view(n, 1, 1, 1) + gray

        x = x.clamp(0.0, 1.0)
        x = (x - mean) / std
        return torch.cat([orig, x], dim=0).to(dtype=dtype)


# ==================== BaseClient锛堜笌 statA(local).py 瀹屽叏涓€鑷达級====================

class BaseClient:
    def __init__(self, dataset, clip_weights, args):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.curr_idx = 0
        self.clip_weights = clip_weights
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


# ==================== BaseCTTAServer锛堜笌 statA(local).py 瀹屽叏涓€鑷达級====================

class BaseCTTAServer:
    def __init__(self, datasets, clip_weights, args, client_class=BaseClient):
        self.num_class, self.feat_dim = clip_weights.shape
        self.clip_weights = clip_weights
        self.clients = OrderedDict(
            [(cid, client_class(dataset, clip_weights, args))
             for (cid, dataset) in datasets.items()])
        self.client_ids = list(datasets.keys())
        self.num_clients = len(datasets)
        self.cid2idx = {cid: idx for idx, cid in enumerate(self.clients)}
        self.idx2cid = {idx: cid for cid, idx in self.cid2idx.items()}
        self.dtype = clip_weights.dtype
        self.device = clip_weights.device
        self.cohort_size = int(self.num_clients * args.part_rate)
        self.num_rounds = sum(client.num_samples for client in self.clients.values())
        self.sync_freq = args.sync_freq

    def evaluate(self):
        total_correct, total_num_samples = 0, 0
        for rnd in tqdm(range(1, self.num_rounds + 1), desc='Zero TTA'):
            if getattr(self.clients[self.client_ids[0]].args, 'max_rounds', -1) > 0 and rnd > self.clients[self.client_ids[0]].args.max_rounds:
                break
            selected_idx = sorted(
                list(torch.randperm(self.num_clients)[:self.cohort_size].numpy()))
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


# ==================== ZeroClient ====================

class ZeroClient(BaseClient):
    """
    Zero 鍩虹嚎瀹㈡埛绔紙ICLR 2024: Test-Time Adaptation with Zero Temperature锛夈€?

    鏍稿績绠楁硶锛堜弗鏍煎榻?zero-master/ttas/zero.py zero() 鏂规硶锛夛細
      1. 瀵规瘡涓祴璇曟牱鏈敓鎴?num_views 涓殢鏈哄寮鸿鍥?
      2. CLIP encode_image 鎻愬彇褰掍竴鍖栫壒寰?[N, D]
      3. confidence_filter锛氫繚鐣欑喌鏈€浣庡墠 gamma 姣斾緥瑙嗗浘
      4. zero-temperature softmax锛坋ps锛? 绱姞鎶曠エ
      5. argmax 寰楀埌鏈€缁堥娴嬶紝骞崇エ鏃?greedy tie-breaking

    娉ㄦ剰锛?
      - Zero 涓嶆敮鎸侀缂撳瓨鐗瑰緛锛屾暟鎹泦蹇呴』杩斿洖鍥剧墖 Tensor
      - 姣忔牱鏈墠鍚?num_views 娆?encode_image锛岃绠楅噺杈冨ぇ浣嗙函鎺ㄧ悊鏃犳搴?
      - 褰撳墠浣跨敤寮犻噺澧炲己璺緞锛岄伩鍏?Tensor 鈫?PIL 杞崲寮€閿€
    """

    def __init__(self, dataset, clip_weights, args):
        super().__init__(dataset, clip_weights, args)

        zero_cfg = args.config.get('zero', {})
        self.num_views = zero_cfg.get('num_views', 64)
        self.gamma     = zero_cfg.get('gamma', 0.3)
        self.use_augmix = zero_cfg.get('augmix', True)

        self.augmenter = AugMixAugmenter(augmix=self.use_augmix)

    @torch.no_grad()
    def evaluate_one(self):
        if self.curr_idx >= self.num_samples:
            return 0, 0

        data, label = self.dataset[self.curr_idx]
        self.curr_idx += 1
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)

        if torch.is_tensor(data):
            if data.dim() == 1 or data.dim() == 2:
                feat = data.to(device=self.device, dtype=self.dtype)
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                feat = F.normalize(feat, dim=1)
                pred = self.predict(feat)
                correct = int(pred == label_int)
                self.num_correct += correct
                return correct, 1
            views = self.augmenter(
                data, num_views=self.num_views,
                device=self.device, dtype=self.dtype)
        else:
            raise TypeError('Zero GPU AugMix expects preprocessed image tensors. Build data loader with CLIP preprocess.')

        # CLIP image features
        img_feats = self.clip_model.encode_image(views)   # [N, D]
        img_feats = F.normalize(img_feats, dim=-1)

        # unscaled logits [N, C]锛坈lip_weights 宸插綊涓€鍖栵級
        l = img_feats @ self.clip_weights.T               # [N, C]
        # 鐢ㄥ浐瀹氭俯搴?100 璁＄畻姒傜巼锛屼粎鐢ㄤ簬 confidence filter 鐨勭喌
        p = (l * 100.0).softmax(dim=1)                   # [N, C]

        # confidence filter锛氫繚鐣欑喌鏈€浣庡墠 gamma 姣斾緥
        # clamp 閬垮厤 log(0) 浜х敓 NaN锛堜笌鍘熺増 zero.py 琛屼负涓€鑷达級
        batch_entropy = -(p * p.clamp(min=1e-8).log()).sum(dim=1)  # [N]
        sorted_idx    = torch.argsort(batch_entropy, descending=False)
        k             = max(1, int(l.size(0) * self.gamma))
        filt_idx      = sorted_idx[:k]
        l_filt        = l[filt_idx]                       # [k, C]

        # zero-temperature softmax + 绱姞鎶曠エ
        zero_temp = torch.finfo(l_filt.dtype).eps
        p_bar = (l_filt / zero_temp).softmax(dim=1).sum(dim=0)  # [C]

        # argmax 棰勬祴
        max_votes, scalar_pred = torch.max(p_bar, dim=-1)
        pred_int = scalar_pred.item()

        # greedy tie-breaking锛堝榻愬師鐗?zero.py锛?
        ties = [pred_int]
        for i in range(p_bar.size(0)):
            if i != pred_int and p_bar[i] == max_votes:
                ties.append(i)
        if len(ties) > 1:
            # 浠庝綆缃俊搴﹀墿浣欒鍥句腑 greedy 鎵剧涓€涓睘浜庡钩绁ㄥ€欓€夌殑棰勬祴
            remaining_l = l[sorted_idx[k:]]              # [remaining, C]
            ties_set = set(ties)
            pred_int = ties[0]                           # 榛樿
            for row in remaining_l:
                candidate = int(row.argmax().item())
                if candidate in ties_set:
                    pred_int = candidate
                    break

        correct = int(pred_int == label_int)
        self.num_correct += correct
        return correct, 1


# ==================== ZeroServer ====================

class ZeroServer(BaseCTTAServer):
    """Zero baseline server with no cross-client communication."""

    def __init__(self, datasets, clip_weights, args):
        super().__init__(datasets, clip_weights, args, client_class=ZeroClient)


# ==================== 涓诲嚱鏁帮紙涓?statA(local).py 缁撴瀯瀹屽叏涓€鑷达級====================

def main():
    args = get_arguments()
    config_path = args.config

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 鍔犺浇 CLIP 妯″瀷锛堜笌 statA(local).py 瀹屽叏涓€鑷达級
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
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    datasets_list = args.datasets.split('/')
    for dataset_name in datasets_list:
        print(f"Processing {dataset_name} dataset.")

        # 璇诲彇閰嶇疆鏂囦欢锛堜笌 tda_runner.py 鐩稿悓鐨勬槧灏勯€昏緫锛?
        _name_map = {
            'VLCS':           'vlcs',
            'TerraIncognita': 'terra',
            'OfficeHome':     'officehome',
            'CIFAR10CFull':   'cifar10c',
            'CIFAR100CFull':  'cifar100c',
        }
        _suffix      = _name_map.get(dataset_name, dataset_name.lower())
        _cfg_file    = os.path.join(config_path, f'zero_{_suffix}.yaml')
        if not os.path.exists(_cfg_file):
            _cfg_file = os.path.join(os.path.dirname(__file__), 'configs', f'zero_{_suffix}.yaml')
        if os.path.exists(_cfg_file):
            with open(_cfg_file, 'r', encoding='utf-8') as _f:
                cfg = yaml.load(_f, Loader=yaml.SafeLoader)
        else:
            cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        args.config = cfg

        # 鍔犺浇鏁版嵁锛堜笌 statA(local).py 瀹屽叏涓€鑷达級
        domain_loaders, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess, separate_domains=True)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.separate_domains:
            # ---- Separate Domain Mode锛堜笌 statA(local).py 瀹屽叏瀵归綈锛?---
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name} - Separate Domain Mode [Zero]")
            print(f"{'='*60}\n")

            domain_results = {}
            total_correct_all = 0
            total_samples_all = 0

            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f"\n--- Evaluating domain: {domain_name} ---")
                dataset = domain_loader.dataset

                # Zero 涓嶆敮鎸佺紦瀛樼壒寰侊紝璺宠繃 cache_features
                total_samples_domain = len(dataset)
                samples_per_client   = total_samples_domain // args.num_clients
                from torch.utils.data import Subset
                client_datasets = OrderedDict()
                rng = np.random.RandomState(args.seed + domain_idx)
                shuffled = rng.permutation(total_samples_domain)
                for i in range(args.num_clients):
                    start_idx = i * samples_per_client
                    end_idx = (start_idx + samples_per_client
                               if i < args.num_clients - 1 else total_samples_domain)
                    client_datasets[f'client_{i}'] = Subset(
                        dataset, shuffled[start_idx:end_idx].tolist())

                print(f"  Created {len(client_datasets)} clients for {domain_name}.")

                server = ZeroServer(client_datasets, clip_weights, args)
                acc, total_correct, total_samples, stats = server.evaluate()

                domain_results[domain_name] = {
                    'accuracy': acc, 'correct': total_correct, 'total': total_samples
                }
                total_correct_all += total_correct
                total_samples_all += total_samples
                print(f"  {domain_name}: {acc * 100:.2f}% ({total_correct}/{total_samples})")

            overall_acc = total_correct_all / total_samples_all
            print(f"\n{'='*60}")
            print(f"Results for {dataset_name} [Zero]:")
            print(f"{'='*60}")
            for domain_name, res in domain_results.items():
                print(f"  {domain_name:20s}: {res['accuracy'] * 100:6.2f}%")
            print(f"  {'Total':20s}: {overall_acc * 100:6.2f}%"
                  f" ({total_correct_all}/{total_samples_all})")
            print(f"{'='*60}\n")

            if args.wandb:
                run_name = f"{dataset_name}_zero_separate"
                run = wandb.init(project="Latte-CTTA", config=cfg,
                                 group=group_name, name=run_name)
                for domain_name, res in domain_results.items():
                    wandb.log({f"{dataset_name}/{domain_name}": res['accuracy'] * 100})
                wandb.log({f"{dataset_name}/Total": overall_acc * 100})
                run.finish()

        else:
            # ---- Collaborative Mode锛堜笌 statA(local).py 瀹屽叏瀵归綈锛?---
            num_domains   = len(domain_loaders)
            total_clients = num_domains * args.num_clients
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name} - Collaborative Mode [Zero]")
            print(f"Total: {total_clients} clients"
                  f" ({num_domains} domains x {args.num_clients} clients)")
            print(f"{'='*60}\n")

            all_client_datasets = OrderedDict()
            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f"Setting up clients for {domain_name}...")
                dataset = domain_loader.dataset

                # Zero 涓嶆敮鎸佺紦瀛樼壒寰?
                total_samples      = len(dataset)
                samples_per_client = total_samples // args.num_clients
                from torch.utils.data import Subset

                if dataset_name in ['CIFAR10CFull', 'CIFAR100CFull']:
                    rng = np.random.RandomState(args.seed + domain_idx)
                    indices_pool = rng.permutation(total_samples).tolist()
                else:
                    rng = np.random.RandomState(args.seed + domain_idx)
                    indices_pool = rng.permutation(total_samples).tolist()

                for i in range(args.num_clients):
                    start_idx = i * samples_per_client
                    end_idx = (start_idx + samples_per_client
                               if i < args.num_clients - 1 else total_samples)
                    all_client_datasets[f'{domain_name}_client_{i}'] = Subset(
                        dataset, indices_pool[start_idx:end_idx])

                print(f"  Created {args.num_clients} clients,"
                      f" each with ~{samples_per_client} samples")

            print(f"\nCreated {len(all_client_datasets)} clients total.")

            if args.wandb:
                run_name = f"{dataset_name}_zero_collaborative"
                run = wandb.init(project="Latte-CTTA", config=cfg,
                                 group=group_name, name=run_name)

            server = ZeroServer(all_client_datasets, clip_weights, args)
            acc, total_correct, total_samples, stats = server.evaluate()

            print(f"\n{'='*60}")
            print(f"Overall Results [Zero]:")
            print(f"{'='*60}")
            print(f"Total Accuracy: {acc * 100:.2f}% ({total_correct}/{total_samples})")
            print(f"{'='*60}\n")

            corruption_results = {}
            for domain_name in domain_loaders.keys():
                corruption_correct = 0
                corruption_total   = 0
                for i in range(args.num_clients):
                    cid = f'{domain_name}_client_{i}'
                    if cid in server.clients:
                        client = server.clients[cid]
                        corruption_correct += client.num_correct
                        corruption_total   += client.num_samples
                if corruption_total > 0:
                    corruption_acc = corruption_correct / corruption_total
                    corruption_results[domain_name] = corruption_acc
                    print(f"  {domain_name:20s}: {corruption_acc * 100:6.2f}%"
                          f" ({corruption_correct}/{corruption_total})")

            if args.wandb:
                wandb.log({f"{dataset_name}/Total": acc * 100})
                for domain_name, domain_acc in corruption_results.items():
                    wandb.log({f"{dataset_name}/{domain_name}": domain_acc * 100})
                run.finish()


if __name__ == '__main__':
    main()
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f'Peak Memory: {peak_mem_gb:.2f} GB')

