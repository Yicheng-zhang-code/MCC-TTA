"""BCA baseline runner."""

import random
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip

from utils import *
from datasets.federated_loader import FederatedDataset, cache_features

# 鏁版嵁闆嗗悕绉?鈫?bca 閰嶇疆鏂囦欢鍚庣紑鏄犲皠
_BCA_NAME_MAP = {
    'VLCS':          'vlcs',
    'TerraIncognita':'terra',
    'CIFAR10CFull':  'cifar10c',
    'CIFAR100CFull': 'cifar100c',
}

def get_bca_config(config_path, dataset_name):
    """Load BCA config, preferring baseline/configs after moving the runner."""
    if config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    suffix = _BCA_NAME_MAP.get(dataset_name, dataset_name.lower())
    search_dirs = [config_path, os.path.join(os.path.dirname(__file__), 'configs')]
    for search_dir in search_dirs:
        bca_file = os.path.join(search_dir, f'bca_{suffix}.yaml')
        if os.path.exists(bca_file):
            with open(bca_file, 'r', encoding='utf-8') as f:
                return yaml.load(f, Loader=yaml.SafeLoader)

    bca_file = os.path.join(config_path, f'bca_{suffix}.yaml')
    if os.path.exists(bca_file):
        with open(bca_file, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)
    # 鍥為€€锛氳蛋鍘熷 get_config_file
    return get_config_file(config_path, dataset_name)


# ==================== BCA Core ====================

class BCA:
    """
    Bayesian Cluster Assignment.
    Maintains class centers and cluster-to-class priors online.
    """
    def __init__(self, init_centers, threshold1, init_count1, threshold2, init_count2):
        self.mu = init_centers.clone()
        self.M = self.mu.size(0)
        self.cluster_to_class_prob = torch.eye(self.M, dtype=torch.float16).cuda()

        self.threshold1 = threshold1
        self.c1 = [init_count1] * self.M
        self.threshold2 = threshold2
        self.c2 = [init_count2] * self.M
        self.tem = 100.0

    def assign_label(self, image_feature):
        """Assign labels and update BCA online."""
        with torch.no_grad():
            # P(x | u_m)
            P_x_um = self.tem * image_feature @ self.mu.t()  # [N, C]

            if image_feature.size(0) > 1:
                # Batch mode: update centers from low-entropy samples.
                batch_entropy = softmax_entropy(P_x_um)
                selected_idx = torch.argsort(batch_entropy, descending=False)[
                    :max(1, int(batch_entropy.size(0) * 0.1))]
                rep_feature = image_feature[selected_idx].mean(0).unsqueeze(0)  # 浠ｈ〃鐗瑰緛
                rep_P_x_um  = P_x_um[selected_idx].mean(0).unsqueeze(0)

                # 鐢ㄤ唬琛ㄧ壒寰佸仛鏇存柊鍒ゆ柇
                s1_rep = torch.softmax(rep_P_x_um, dim=1)
                f1_rep = s1_rep @ self.cluster_to_class_prob
                prob_max_rep, pred_rep = torch.max(f1_rep, dim=1)
                if prob_max_rep.item() > self.threshold1:
                    self.update_centers(rep_feature, pred_rep)
                if prob_max_rep.item() > self.threshold2:
                    self.update_prior(f1_rep, pred_rep)

                P_all = self.tem * image_feature @ self.mu.t()  # [N, C]
                s1_all = torch.softmax(P_all, dim=1)
                f1_all = s1_all @ self.cluster_to_class_prob
                preds = torch.argmax(f1_all, dim=1)  # [N]
                return preds  # 杩斿洖 tensor锛屼緵鎵归噺 correct 缁熻

            s1 = torch.softmax(P_x_um, dim=1)
            f1 = s1 @ self.cluster_to_class_prob
            prob_max, pred = torch.max(f1, dim=1)

            if prob_max.item() > self.threshold1:
                self.update_centers(image_feature, pred)
            if prob_max.item() > self.threshold2:
                self.update_prior(f1, pred)

        return int(pred.item())

    def update_centers(self, image_feature, pred):
        """Update cluster centers with EMA-style counts."""
        with torch.no_grad():
            p = int(pred.item())
            self.mu[p] = self.c1[p] * self.mu[p] + image_feature.squeeze(0)
            self.c1[p] += 1
            self.mu[p] = self.mu[p] / self.c1[p]
            self.mu[p] = self.mu[p] / self.mu[p].norm()

    def update_prior(self, soft_prob, pred):
        """Update cluster-to-class prior with EMA-style counts."""
        with torch.no_grad():
            p = int(pred.item())
            self.cluster_to_class_prob[p] = (
                self.c2[p] * self.cluster_to_class_prob[p] + soft_prob
            ) / (self.c2[p] + 1)
            self.c2[p] += 1


# ==================== 鍙傛暟瑙ｆ瀽锛堜笌 statA(local).py 瀹屽叏涓€鑷达級====================

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True,
                        help='settings of BCA on specific dataset in yaml format.')
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
    parser.add_argument('--cache-features', dest='cache_features', action='store_true')
    parser.add_argument('--separate-domains', dest='separate_domains', action='store_true')
    parser.add_argument('--global-mode', dest='global_mode', action='store_true',
                        help='Enable cross-client BCA center aggregation (FedAvg on mu)')
    return parser.parse_args()


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

    def is_done(self):
        return self.curr_idx >= self.num_samples


# ==================== BCALocalClient ====================

class BCALocalClient(BaseClient):
    """BCA local baseline client."""

    def __init__(self, dataset, clip_weights, args):
        super().__init__(dataset, clip_weights, args)

        bca_cfg = args.config.get('bca', {})
        threshold1  = float(bca_cfg.get('threshold1',  0.5))
        init_count1 = float(bca_cfg.get('init_count1', 1.0))
        threshold2  = float(bca_cfg.get('threshold2',  0.9))
        init_count2 = float(bca_cfg.get('init_count2', 1.0))
        self.batch_size = int(bca_cfg.get('batch_size', 64))

        init_centers = clip_weights.clone().cuda().half()  # [C, d]
        self.bca = BCA(init_centers, threshold1, init_count1, threshold2, init_count2)

        self.feat_buf   = []   # list of [1, d] fp16
        self.label_buf  = []   # list of int
        self.zs_correct_buf = 0  # 鏈?batch 鐨?ZS 涓存椂 correct

    def _encode(self, data):
        dataset = self.dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        is_cached = hasattr(dataset, 'is_cached_features') and dataset.is_cached_features

        if data.dim() == 3 or (data.dim() == 4 and data.shape[0] == 1):
            if data.dim() == 3:
                data = data.unsqueeze(0)
            data = data.to(device=self.device)
            with torch.no_grad():
                feat = self.clip_model.encode_image(data)
                feat = feat.to(dtype=self.dtype)
                feat = F.normalize(feat, dim=1)
        else:
            feat = data.to(device=self.device, dtype=self.dtype)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            if not is_cached:
                feat = F.normalize(feat, dim=1)
        return feat

    def _flush_batch(self):
        """Flush buffered samples through BCA and update cached correctness."""
        n = len(self.feat_buf)
        if n == 0:
            return

        batch = torch.cat(self.feat_buf, dim=0)  # [N, d] fp16 cuda
        preds = self.bca.assign_label(batch)     # 杩斿洖 [N] tensor

        labels = torch.tensor(self.label_buf, device=preds.device)
        bca_correct = int((preds == labels).sum().item())

        # 淇 num_correct锛氭挙閿€ ZS 涓存椂鍊硷紝鎹㈡垚 BCA 缁撴灉
        self.num_correct -= self.zs_correct_buf
        self.num_correct += bca_correct

        self.feat_buf.clear()
        self.label_buf.clear()
        self.zs_correct_buf = 0

    @torch.no_grad()
    def evaluate_one(self):
        """Evaluate one sample and flush BCA when the buffer is full."""
        if self.curr_idx >= self.num_samples:
            return 0, 0

        data, label = self.dataset[self.curr_idx]
        self.curr_idx += 1

        feat = self._encode(data)                    # [1, d]
        feat_half = feat.cuda().half()
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)

        if self.batch_size <= 1:
            # 閫愭牱鏈ā寮忥細鐩存帴 BCA 鎺ㄦ柇+鏇存柊锛堝師璁烘枃 init_count1=20000 闃插穿濉岋級
            pred = self.bca.assign_label(feat_half)
            correct = int(pred == label_int)
            self.num_correct += correct
            return correct, 1

        # batch buffer 妯″紡锛歓S 涓存椂棰勬祴锛屾敀婊″悗鎵归噺 BCA 淇
        zs_logits  = 100.0 * feat @ self.clip_weights.T  # [1, C]
        zs_pred    = int(zs_logits.argmax(dim=1).item())
        zs_correct = int(zs_pred == label_int)
        self.num_correct    += zs_correct
        self.zs_correct_buf += zs_correct

        self.feat_buf.append(feat_half)
        self.label_buf.append(label_int)

        if len(self.feat_buf) >= self.batch_size:
            self._flush_batch()

        return zs_correct, 1

    def flush_remaining(self):
        """Flush leftover samples smaller than one batch."""
        self._flush_batch()


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
        for rnd in tqdm(range(1, self.num_rounds + 1), desc='BCA-Local TTA'):
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
        self.syncronize()
        final_correct = sum(client.num_correct for client in self.clients.values())
        final_total   = sum(client.num_samples for client in self.clients.values())
        stats = {cid: client.num_correct / client.num_samples
                 for (cid, client) in self.clients.items()}
        return final_correct / final_total, final_correct, final_total, stats

    def syncronize(self):
        pass  # Local baseline: no cross-client communication.


# ==================== BCALocalServer ====================

class BCALocalServer(BaseCTTAServer):
    """BCA local baseline server."""
    def __init__(self, datasets, clip_weights, args,
                 client_class=BCALocalClient):
        super().__init__(datasets, clip_weights, args, client_class)

    def syncronize(self):
        """Flush remaining buffers without cross-client communication."""
        for client in self.clients.values():
            if hasattr(client, 'flush_remaining'):
                client.flush_remaining()


# ==================== BCAGlobalServer ====================

class BCAGlobalServer(BCALocalServer):
    """BCA global baseline server using FedAvg over cluster centers."""
    def syncronize(self):
        # 鍏堝埛鏂板悇瀹㈡埛绔墿浣?buffer
        for client in self.clients.values():
            if hasattr(client, 'flush_remaining'):
                client.flush_remaining()

        # 鑱氬悎锛氬姣忎釜绫诲埆锛屾寜鍚勫鎴风鐨勬牱鏈暟鍔犳潈骞冲潎 mu
        clients = list(self.clients.values())
        total_samples = sum(c.num_samples for c in clients)
        if total_samples == 0:
            return

        # [num_clients, C, d]
        mu_stack = torch.stack([c.bca.mu for c in clients], dim=0)  # [K, C, d]
        weights = torch.tensor(
            [c.num_samples / total_samples for c in clients],
            dtype=mu_stack.dtype, device=mu_stack.device
        ).view(-1, 1, 1)  # [K, 1, 1]
        mu_avg = (mu_stack * weights).sum(dim=0)  # [C, d]
        mu_avg = F.normalize(mu_avg, dim=1)  # [C, d]

        # 骞挎挱缁欐墍鏈夊鎴风
        for client in clients:
            client.bca.mu = mu_avg.clone()
            client.bca.c1 = [client.bca.c1[i] for i in range(client.bca.M)]


# ==================== 涓诲嚱鏁帮紙涓?statA(local).py 缁撴瀯瀹屽叏涓€鑷达級====================

def main():
    args = get_arguments()
    config_path = args.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 鍔犺浇 CLIP 妯″瀷
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

    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")

        cfg = get_bca_config(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        args.config = cfg

        domain_loaders, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess, separate_domains=True)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.separate_domains:
            # ---- Separate Domain Mode ----
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name} - Separate Domain Mode [BCA-Local]")
            print(f"{'='*60}\n")

            domain_results = {}
            total_correct_all = 0
            total_samples_all = 0

            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f"\n--- Evaluating domain: {domain_name} ---")
                dataset = domain_loader.dataset

                if args.cache_features:
                    print(f"Pre-caching CLIP features for {domain_name}...")
                    cache_name = f"{dataset_name}_{domain_name}_{args.backbone.replace('/', '_')}"
                    dataset = cache_features(dataset, clip_model, device=device,
                                            cache_dir='./cached_features',
                                            dataset_name=cache_name)

                total_samples_domain = len(dataset)
                samples_per_client = total_samples_domain // args.num_clients
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

                ServerClass = BCAGlobalServer if getattr(args, 'global_mode', False) else BCALocalServer
                server = ServerClass(client_datasets, clip_weights, args,
                                     client_class=BCALocalClient)
                acc, total_correct, total_samples, stats = server.evaluate()

                domain_results[domain_name] = {
                    'accuracy': acc, 'correct': total_correct, 'total': total_samples
                }
                total_correct_all += total_correct
                total_samples_all += total_samples
                print(f"  {domain_name}: {acc * 100:.2f}% ({total_correct}/{total_samples})")

            overall_acc = total_correct_all / total_samples_all
            print(f"\n{'='*60}")
            print(f"Results for {dataset_name} [BCA-Local]:")
            print(f"{'='*60}")
            for domain_name, res in domain_results.items():
                print(f"  {domain_name:20s}: {res['accuracy'] * 100:6.2f}%")
            print(f"  {'Total':20s}: {overall_acc * 100:6.2f}%"
                  f" ({total_correct_all}/{total_samples_all})")
            print(f"{'='*60}\n")

            if args.wandb:
                run_name = f"{dataset_name}_bca_local_separate"
                run = wandb.init(project="MCC-TTA-Baselines", config=cfg,
                                 group=group_name, name=run_name)
                for domain_name, res in domain_results.items():
                    wandb.log({f"{dataset_name}/{domain_name}": res['accuracy'] * 100})
                wandb.log({f"{dataset_name}/Total": overall_acc * 100})
                run.finish()

        else:
            # ---- Collaborative Mode ----
            num_domains = len(domain_loaders)
            total_clients = num_domains * args.num_clients
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name} - Collaborative Mode [BCA-Local]")
            print(f"Total: {total_clients} clients"
                  f" ({num_domains} domains x {args.num_clients} clients)")
            print(f"{'='*60}\n")

            all_client_datasets = OrderedDict()
            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f"Setting up clients for {domain_name}...")
                dataset = domain_loader.dataset

                if args.cache_features:
                    cache_name = f"{dataset_name}_{domain_name}_{args.backbone.replace('/', '_')}"
                    dataset = cache_features(dataset, clip_model, device=device,
                                            cache_dir='./cached_features',
                                            dataset_name=cache_name)

                total_samples = len(dataset)
                samples_per_client = total_samples // args.num_clients
                from torch.utils.data import Subset

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
                run_name = f"{dataset_name}_bca_local_collaborative"
                run = wandb.init(project="MCC-TTA-Baselines", config=cfg,
                                 group=group_name, name=run_name)

            ServerClass = BCAGlobalServer if getattr(args, 'global_mode', False) else BCALocalServer
            server = ServerClass(all_client_datasets, clip_weights, args,
                                 client_class=BCALocalClient)
            acc, total_correct, total_samples, stats = server.evaluate()

            print(f"\n{'='*60}")
            print(f"Overall Results [BCA-Local]:")
            print(f"{'='*60}")
            print(f"Total Accuracy: {acc * 100:.2f}% ({total_correct}/{total_samples})")
            print(f"{'='*60}\n")

            corruption_results = {}
            for domain_name in domain_loaders.keys():
                corruption_correct = 0
                corruption_total = 0
                for i in range(args.num_clients):
                    cid = f'{domain_name}_client_{i}'
                    if cid in stats:
                        client = server.clients[cid]
                        corruption_correct += client.num_correct
                        corruption_total += client.num_samples
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
    

