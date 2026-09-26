# -*- coding: utf-8 -*-
"""TDA baseline runner with local/global modes controlled by share_cache."""

import random
import argparse
import operator
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import wandb
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
import math

import torch
import torch.nn.functional as F
import numpy as np
import clip

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
    parser.add_argument('--partition', dest='partition', type=str, default='iid',
                        choices=['iid', 'domain', 'corruption'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cache-features', dest='cache_features', action='store_true')
    parser.add_argument('--separate-domains', dest='separate_domains', action='store_true')
    args = parser.parse_args()
    return args


# ==================== BaseClient ====================

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


# ==================== BaseCTTAServer ====================

class BaseCTTAServer:
    def __init__(self, datasets, clip_weights, args, client_class=BaseClient):
        self.num_class, self.feat_dim = clip_weights.shape
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


# ==================== TDAClient ====================

class TDAClient(BaseClient):
    """TDA baseline client."""
    def __init__(self, dataset, clip_weights, args):
        super(TDAClient, self).__init__(dataset, clip_weights, args)

        cfg = args.config.get('tda', {})
        self.pos_capacity = cfg.get('pos_capacity', 3)
        self.neg_capacity = cfg.get('neg_capacity', 2)
        self.pos_alpha    = cfg.get('pos_alpha', 2.0)
        self.pos_beta     = cfg.get('pos_beta', 5.0)
        self.neg_alpha    = cfg.get('neg_alpha', 0.117)
        self.neg_beta     = cfg.get('neg_beta', 1.0)
        self.ent_lower    = cfg.get('entropy_lower', 0.2)
        self.ent_upper    = cfg.get('entropy_upper', 0.5)
        self.mask_lower   = cfg.get('mask_lower', 0.03)
        self.mask_upper   = cfg.get('mask_upper', 1.0)
        self.neg_enabled  = cfg.get('neg_enabled', True)

        self.pos_cache = {}  # {class_idx: [[feature, loss], ...]}
        self.neg_cache = {}  # {class_idx: [[feature, loss, prob_map], ...]}

        # pos_keys:   [feat_dim, num_class * pos_capacity]
        # pos_values: [num_class * pos_capacity, num_class]
        _C, _D, _K = self.num_class, self.feat_dim, self.pos_capacity
        self._pos_tensor = torch.zeros(_C, _K, _D,
                                       device=self.device, dtype=self.dtype)  # [C, K, D]
        self._pos_count  = torch.zeros(_C, dtype=torch.long, device=self.device)
        self._pos_loss   = torch.full((_C, _K), float('inf'),
                                      device=self.device, dtype=torch.float32)  # 姣忔潯鐩?loss
        # 蹇€?logits 鐢ㄧ殑灞曞钩鐭╅樀锛堟瘡娆′粠 _pos_tensor 閲嶅缓锛宻hare_cache 鏃惰嚜鍔ㄥ悓姝ワ級
        self._pos_keys   = None
        self._pos_values = None
        self._pos_values_fixed = F.one_hot(
            torch.arange(_C, device=self.device).repeat_interleave(_K),
            num_classes=_C).to(dtype=self.dtype)           # [C*K, C]

        self.max_entropy = math.log2(self.num_class)

    def _get_entropy(self, loss):
        return float(loss / self.max_entropy)

    def _update_cache(self, cache, pred, features_loss, shot_capacity, include_prob_map=False):
        """Update TDA positive or negative cache."""
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

        # 鍚屾鏇存柊棰勫垎閰?pos_tensor锛堜粎姝ｇ紦瀛橈紝璐熺紦瀛樹笉鐢級
        if not include_prob_map:
            feat = features_loss[0]   # [1, D]
            loss = float(features_loss[1])
            c    = pred
            cnt  = self._pos_count[c].item()
            if cnt < self.pos_capacity:
                # 妲芥湭婊★紝鐩存帴鍐欏叆
                self._pos_tensor[c, cnt] = feat.squeeze(0)
                self._pos_loss[c, cnt]   = loss
                self._pos_count[c]      += 1
            else:
                max_idx = int(self._pos_loss[c].argmax().item())
                if loss < self._pos_loss[c, max_idx].item():
                    self._pos_tensor[c, max_idx] = feat.squeeze(0)
                    self._pos_loss[c, max_idx]   = loss

    def _compute_cache_logits(self, image_features, cache, alpha, beta, neg_mask_thresholds=None):
        """Compute TDA cache logits."""
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0).to(image_features.device)
        target_dtype = image_features.dtype
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) &
                             (cache_values < neg_mask_thresholds[1]))
                            .type(torch.int8)).to(device=image_features.device, dtype=target_dtype)
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64),
                                      num_classes=self.num_class)
                            ).to(device=image_features.device, dtype=target_dtype)

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

    def _build_pos_cache_tensors(self):
        """Build flattened positive cache tensors."""
        C, K, D = self._pos_tensor.shape
        keys = self._pos_tensor.reshape(C * K, D).T.contiguous()  # [D, C*K]
        return keys, self._pos_values_fixed                         # values 棰勬瀯寤猴紝鏃犻渶閲嶅缓

    def _compute_pos_logits_fast(self, image_features):
        """Compute positive cache logits with tensor operations."""
        if not self.pos_cache:
            return None
        keys, values = self._build_pos_cache_tensors()
        # 绌烘Ы锛堝叏闆跺垪锛夌殑 affinity 鑷劧涓?0锛宔xp(-beta*(1-0))鈮?锛屼笉褰卞搷缁撴灉
        affinity = image_features @ keys                    # [1, C*K]
        cache_logits = ((-1) * (self.pos_beta - self.pos_beta * affinity)).exp() @ values  # [1, C]
        return self.pos_alpha * cache_logits

    @torch.no_grad()
    def predict(self, image_feature):
        # 1. 鑾峰彇 CLIP 棰勬祴锛岃繑鍥為『搴忥細(clip_logits, pred, proba, entropy)
        clip_logits, pred, prob_map, loss = get_clip_logits(image_feature, self.clip_weights)

        if torch.is_tensor(loss):
            loss = loss.item()

        prop_entropy = self._get_entropy(loss)

        self._update_cache(self.pos_cache, pred, [image_feature, loss], self.pos_capacity)

        if self.neg_enabled and self.ent_lower < prop_entropy < self.ent_upper:
            self._update_cache(self.neg_cache, pred, [image_feature, loss, prob_map],
                               self.neg_capacity, include_prob_map=True)

        # 4. 璁＄畻 final_logits
        final_logits = clip_logits.clone()
        if self.pos_cache:
            final_logits = final_logits + self._compute_pos_logits_fast(image_feature)
        if self.neg_enabled and self.neg_cache:
            final_logits = final_logits - self._compute_cache_logits(
                image_feature, self.neg_cache, self.neg_alpha, self.neg_beta,
                neg_mask_thresholds=(self.mask_lower, self.mask_upper))

        return final_logits.argmax(dim=1).item()


# ==================== TDAServer ====================

class TDAServer(BaseCTTAServer):
    """TDA baseline server."""
    def __init__(self, datasets, clip_weights, args):
        super(TDAServer, self).__init__(datasets, clip_weights, args, client_class=TDAClient)


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
    
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f'Processing {dataset_name} dataset.')
        
        # TDA uses tda_*.yaml instead of the MCC-TTA config mapping.
        _tda_name_map = {
            'VLCS': 'vlcs',
            'TerraIncognita': 'terra',
            'OfficeHome': 'officehome',
            'CIFAR10CFull': 'cifar10c',
            'CIFAR100CFull': 'cifar100c',
        }
        _tda_suffix = _tda_name_map.get(dataset_name, dataset_name.lower())
        _tda_config_file = os.path.join(config_path, f'tda_{_tda_suffix}.yaml')
        if not os.path.exists(_tda_config_file):
            _tda_config_file = os.path.join(os.path.dirname(__file__), 'configs', f'tda_{_tda_suffix}.yaml')
        if os.path.exists(_tda_config_file):
            with open(_tda_config_file, 'r', encoding='utf-8') as _f:
                cfg = yaml.load(_f, Loader=yaml.SafeLoader)
        else:
            cfg = get_config_file(config_path, dataset_name)
        print('\nRunning dataset configurations:')
        print(cfg, '\n')
        args.config = cfg
        
        tda_cfg = cfg.get('tda', {})
        share_cache = tda_cfg.get('share_cache', False)
        neg_enabled  = tda_cfg.get('neg_enabled', True)

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

                server = TDAServer(client_datasets, clip_weights, args)

                # 鍩熷唴鍏变韩缂撳瓨
                if share_cache:
                    _clients = list(server.clients.values())
                    if len(_clients) > 1:
                        _c0 = _clients[0]
                        for _c in _clients[1:]:
                            _c.pos_cache     = _c0.pos_cache
                            # 鍏变韩棰勫垎閰?tensor锛堜繚璇佸啓鍏ュ悓姝ワ級
                            _c._pos_tensor   = _c0._pos_tensor
                            _c._pos_count    = _c0._pos_count
                            _c._pos_loss     = _c0._pos_loss
                            if neg_enabled:
                                _c.neg_cache = _c0.neg_cache
                        _neg_info = '& neg_cache' if neg_enabled else '(neg_cache 鏈叡浜?'
                        print(f'  [share_cache] {len(_clients)} clients share pos_cache {_neg_info} (separate-domains mode).')

                if args.wandb:
                    run_name = f'{dataset_name}_{domain_name}'
                    run = wandb.init(project='TDA-CTTA', config=cfg, group=group_name, name=run_name)

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
                
                # 鎵€鏈夋暟鎹泦缁熶竴浣跨敤闅忔満鍒嗛厤锛屾瘡涓?domain 浣跨敤涓嶅悓闅忔満绉嶅瓙
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
                run = wandb.init(project='TDA-CTTA', config=cfg, group=group_name, name=run_name)

            server = TDAServer(all_client_datasets, clip_weights, args)

            if share_cache:
                _clients = list(server.clients.values())
                if len(_clients) > 1:
                    _c0 = _clients[0]
                    for _c in _clients[1:]:
                        _c.pos_cache     = _c0.pos_cache
                        # 鍏变韩棰勫垎閰?tensor锛堜繚璇佸啓鍏ュ悓姝ワ級
                        _c._pos_tensor   = _c0._pos_tensor
                        _c._pos_count    = _c0._pos_count
                        _c._pos_loss     = _c0._pos_loss
                        if neg_enabled:
                            _c.neg_cache = _c0.neg_cache
                    _neg_info = '& neg_cache' if neg_enabled else '(neg_cache 鏈叡浜?'
                    print(f'  [share_cache] {len(_clients)} clients share pos_cache {_neg_info} (global mode).')

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

