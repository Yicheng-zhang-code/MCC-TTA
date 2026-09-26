"""
StatA Local Baseline Runner - 联邦对比基线（Local版）

Local StatA baseline: each client runs independently without communication.
The runner follows the shared baseline data partition and CLI conventions.
"""

import random
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import wandb
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import clip

from utils import *
from datasets.federated_loader import FederatedDataset, cache_features


def get_stata_config_file(config_path, dataset_name):
    """Load StatA configs from a directory or an explicit yaml path."""
    if config_path.endswith((".yaml", ".yml")):
        config_file = config_path
    else:
        config_mapping = {
            "VLCS": "stata_vlcs.yaml",
            "TerraIncognita": "stata_terra.yaml",
            "CIFAR10CFull": "stata_cifar10c.yaml",
            "CIFAR100CFull": "stata_cifar100c.yaml",
        }
        config_name = config_mapping.get(dataset_name, f"stata_{dataset_name.lower()}.yaml")
        config_file = os.path.join(config_path, config_name)
        if not os.path.exists(config_file):
            config_file = os.path.join(os.path.dirname(__file__), "configs", config_name)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def get_zero_shot_logits(query_features, query_labels, clip_prototypes):
    clip_logits = 100 * query_features @ clip_prototypes
    return clip_logits.squeeze()


def build_affinity_matrix(query_features, n_neighbors):
    device = query_features.device
    num_samples = query_features.size(0)
    affinity = query_features.matmul(query_features.T).cpu()
    knn_index = affinity.topk(n_neighbors + 1, -1, largest=True).indices[:, 1:]
    row_indices = torch.arange(num_samples).unsqueeze(1).repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices].to(device)
    return torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]).to(device),
        values,
        size=(num_samples, num_samples),
        device=device,
    )


class Gaussian(nn.Module):
    def __init__(self, mu, cov):
        super().__init__()
        self.mu = mu.clone()
        self.cov = cov.clone()

    def forward(self, x, no_exp=False):
        chunk_size = 2500
        num_samples = x.shape[0]
        num_classes = self.mu.shape[0]
        likelihoods = torch.empty((num_samples, num_classes), dtype=x.dtype, device=x.device)

        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            likelihoods[start_idx:end_idx] = -0.5 * (
                ((x[start_idx:end_idx][:, None, :] - self.mu[None, :, 0, :]) ** 2)
                * (1 / self.cov[None, :, :])
            ).sum(dim=2)

        if not no_exp:
            likelihoods = torch.exp(likelihoods)
        return likelihoods

    def set_cov(self, cov):
        self.cov = cov

    def set_mu(self, mu):
        self.mu = mu


def update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors, sigma, max_iter=5):
    for _ in range(max_iter):
        intermediate = likelihoods.clone()
        intermediate += lambda_laplacian * (50 / (n_neighbors * 2)) * (W.T @ z + (W @ z))
        intermediate -= 0.5 * sigma.log().sum(dim=1).unsqueeze(0)
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        intermediate = (y_hat ** lambda_y_hat) * torch.exp(1 / 50 * intermediate)
        z = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


def update_mu(adapter, query_features, z, beta, init_prototypes):
    mu = torch.einsum("ij,ik->jk", z, query_features)
    mu /= torch.sum(z, dim=0).unsqueeze(-1)
    mu = mu.unsqueeze(1)
    mu /= mu.norm(dim=-1, keepdim=True)
    mu = beta.unsqueeze(-1).unsqueeze(-1) * mu + (1 - beta).unsqueeze(-1).unsqueeze(-1) * init_prototypes
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def update_cov(adapter, query_features, z, beta, init_prototypes, init_covariance):
    chunk_size = 2500
    cov = None
    for start_idx in range(0, z.size(0), chunk_size):
        end_idx = min(start_idx + chunk_size, z.size(0))
        query_features_chunk = query_features[start_idx:end_idx]
        weighted_sum = (
            (query_features_chunk[:, None, :] - adapter.mu[None, :, 0, :]) ** 2
            * z[start_idx:end_idx, :, None]
        ).sum(dim=0)
        cov = weighted_sum if cov is None else cov + weighted_sum

    cov /= z.sum(dim=0)[:, None]
    delta_mu = (init_prototypes - adapter.mu).squeeze()
    diagonal_result = torch.diagonal(torch.bmm(delta_mu.unsqueeze(2), delta_mu.unsqueeze(1)), dim1=1, dim2=2)
    return beta.unsqueeze(-1) * cov + (1 - beta).unsqueeze(-1) * (init_covariance + diagonal_result)


def init_cov(clip_prototypes, query_features, z):
    chunk_size = 2500
    cov = None
    for start_idx in range(0, z.size(0), chunk_size):
        end_idx = min(start_idx + chunk_size, z.size(0))
        query_features_chunk = query_features[start_idx:end_idx]
        chunk_result = torch.einsum(
            "ij,ijk->k",
            z[start_idx:end_idx, :],
            (query_features_chunk[:, None, :] - clip_prototypes[None, :, 0, :]) ** 2,
        )
        cov = chunk_result if cov is None else cov + chunk_result
        cov /= z.size(0)
    return cov


def update_beta(z, alpha, soft=False):
    if soft:
        sum_z = torch.sum(z, dim=0)
    else:
        predicted_classes = torch.argmax(z, dim=1)
        sum_z = torch.bincount(predicted_classes, minlength=z.size(1))
    return sum_z / (alpha + sum_z + 1e-12)


def StatA_solver(query_features, query_labels, clip_prototypes, alpha=1, soft_beta=False, lambda_y_hat=1, lambda_laplacian=1, n_neighbors=3, max_iter=10):
    query_labels = query_labels.cuda().float()
    clip_prototypes = clip_prototypes.cuda().float()
    query_features = query_features.cuda().float()

    zs_logits = get_zero_shot_logits(query_features, query_labels, clip_prototypes)
    y_hat = F.softmax(zs_logits, dim=1)
    z = y_hat.clone()

    mu = clip_prototypes.permute(2, 0, 1)
    cov = init_cov(clip_prototypes.permute(2, 0, 1), query_features, z)
    cov = cov.unsqueeze(0).repeat(y_hat.size(-1), 1)
    init_covariance = cov

    adapter = Gaussian(mu=mu, cov=cov).cuda()
    W = build_affinity_matrix(query_features.float(), n_neighbors)

    for k in range(max_iter + 1):
        likelihoods = adapter(query_features, no_exp=True)
        z = update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors, adapter.cov)
        if k == max_iter:
            break
        beta = update_beta(z, alpha, soft=soft_beta)
        mu = update_mu(adapter, query_features, z, beta, clip_prototypes.permute(2, 0, 1))
        adapter.set_mu(mu)
        cov = update_cov(adapter, query_features, z, beta, clip_prototypes.permute(2, 0, 1), init_covariance)
        adapter.set_cov(cov)

    return y_hat.cpu(), z.cpu()


# ==================== Argument parsing ====================

def get_arguments():
    """Parse the common baseline evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, 
                        help='settings of StatA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', 
                        help='Whether you want to log to wandb.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, 
                        help='Datasets to process, separated by a slash (/).')
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', 
                        help='Path to the datasets directory.')
    parser.add_argument('--backbone', dest='backbone', type=str,
                        choices=['RN50', 'ViT-B/16'], required=True,
                        help='CLIP model backbone to use.')
    parser.add_argument('--num-clients', dest='num_clients', type=int, default=10)
    parser.add_argument('--part-rate', dest='part_rate', type=float, default=1.0)
    parser.add_argument('--sync-freq', dest='sync_freq', type=int, default=100,
                        help='StatA batch window size: run StatA every N rounds per client.')
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
        for rnd in tqdm(range(1, self.num_rounds + 1), desc='StatA-Local TTA'):
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
        # 处理最后一个不足 sync_freq 的尾部窗口
        self.syncronize()
        # 注意：total_correct 累积的是 zero-shot 临时预测值。
        # syncronize() 已用 StatA 结果修正了每个 client.num_correct。
        # 因此最终精度从 client.num_correct 重新汇总，确保是 StatA 结果。
        final_correct = sum(client.num_correct for client in self.clients.values())
        final_total   = sum(client.num_samples for client in self.clients.values())
        stats = {cid: client.num_correct / client.num_samples
                 for (cid, client) in self.clients.items()}
        return final_correct / final_total, final_correct, final_total, stats

    def syncronize(self):
        pass


# ==================== StatALocalClient ====================

class StatALocalClient(BaseClient):
    """
    StatA Local 基线客户端。

    行为：
      - 每个样本到来时，先用 zero-shot CLIP 做临时预测（实时 correct 计数）。
      - 同时将特征和标签存入本地 buffer。
      - 每次 synchronize() 时，用积累的 buffer 独立运行 StatA_solver，
        用 StatA 的软标签 z 重新计算这批样本的 correct，修正 num_correct。
      - 客户端之间零通信，完全独立。
    """

    def __init__(self, dataset, clip_weights, args):
        super().__init__(dataset, clip_weights, args)

        # StatA 超参数从 config['stata'] 读取，提供合理默认值
        stata_cfg = args.config.get('stata', {})
        self.stata_alpha        = stata_cfg.get('alpha', 1.0)
        self.stata_soft_beta    = stata_cfg.get('soft_beta', False)
        self.stata_lambda_y_hat = stata_cfg.get('lambda_y_hat', 1.0)
        self.stata_lambda_lap   = stata_cfg.get('lambda_laplacian', 1.0)
        self.stata_n_neighbors  = stata_cfg.get('n_neighbors', 3)
        self.stata_max_iter     = stata_cfg.get('max_iter', 10)

        # StatA_solver 内部结构分析：
        #   get_zero_shot_logits: [N,d] @ clip_prototypes → 需要 clip_prototypes=[d,C]
        #   mu init: clip_prototypes.permute(2,0,1) → 需要 clip_prototypes=[d,C,1]
        # 两者要求不同，但 StatA_solver 只接受一个输入。
        # 正确格式是 [d, C, 1]：get_zero_shot_logits 里 [N,d]@[d,C,1]
        # PyTorch matmul 广播规则：[N,d] @ [d,C,1] → 先把[N,d]看成[N,d]，
        # [d,C,1] 需要 squeeze 成 [d,C] 再做 @，但实际 @ 不会自动 squeeze。
        # 唯一正确解：传 [d, C, 1]，StatA_solver 内部 get_zero_shot_logits
        # 里的 @ 会把 [N,d]@[d,C,1] 当 batch matmul → 出错。
        # 结论：需要在调用前 squeeze，即传 [d, C] 给 get_zero_shot_logits，
        # 同时 mu init 用 unsqueeze。
        # 最简单的解法：直接看 StatA_solver 里 get_zero_shot_logits 调用时
        # clip_prototypes 的实际期望格式 = [d, C, 1]，因为后面有 .squeeze()。
        # [N,d] @ [d,C,1]：PyTorch 会把最后两维做矩阵乘：[d] @ [C,1] → 错误。
        # 真正正确：传 [feat_dim, num_class, 1]，squeeze() 去掉尾部1
        # 但 @ 不支持，所以原版调用时实际传的是什么？
        # → 原版传的就是 [feat_dim, num_class, 1]，@ 后得到 [N, num_class, 1]，
        #   squeeze() 变 [N, num_class]。这需要 broadcast：
        #   [N, d] @ [d, C, 1]：PyTorch matmul 把最后两维视为矩阵：
        #   左边最后两维=[N最后取1行,d]，右边=[d,C]... 不对。
        # 正确解：clip_prototypes 传 [d, C, 1]，@ 做 [N,d]x[d,C,1]:
        #   unsqueeze [N,d] → [N,1,d]，@ [d,C,1] → batch不匹配
        # 最终结论：原版期望传入 [feat_dim, num_class, 1]，
        # query_features [N,feat_dim] @ [feat_dim, num_class, 1]:
        # PyTorch treats as: [..., n, m] @ [..., m, p]
        # 左: [N, feat_dim], 右: [feat_dim, num_class, 1]
        # → 右边有3维，左边有2维，广播后左变[1,N,feat_dim]，右[feat_dim,num_class,1]不match
        # 唯一能work的：clip_prototypes = [feat_dim, num_class]（2D）
        # permute(2,0,1) 要求3D，所以内部用时再 unsqueeze。
        # 结论：传 [feat_dim, num_class, 1]，get_zero_shot_logits 里 squeeze 掉最后维先。
        # 但代码里没有 squeeze，所以原版调用者一定传的是恰好能让@成立的格式。
        # 设 clip_prototypes shape = (A, B, C)，@ 规则最后两维矩阵乘：
        # [N, feat_dim] @ (A,B,C)：左broadcast到(1,N,feat_dim)，右(A,B,C)
        # 需要 feat_dim==B，结果(A,N,C)，squeeze→(A,N)如果C=1
        # 但logits应该是[N,num_class]，不是[A,N]
        # 唯一合理：A=1, B=feat_dim, C=num_class → shape=(1,feat_dim,num_class)
        # 则 [N,feat_dim]@[1,feat_dim,num_class] broadcast→[1,N,num_class], squeeze→[N,num_class] ✓
        # permute(2,0,1) on [1,feat_dim,num_class] → [num_class,1,feat_dim] ✓ (mu shape)
        # clip_weights [C, d] → unsqueeze(0) → [1, C, d] → permute(0,2,1) → [1, d, C]
        # StatA_solver 期望 clip_prototypes 形状为 [1, feat_dim, num_class]
        # 经实测：[1,d,C] 是唯一能让 StatA_solver 完整运行的格式
        #   get_zero_shot_logits: [N,d] @ [1,d,C] -> [1,N,C] -> squeeze -> [N,C] ✓
        #   mu init: permute(2,0,1) on [1,d,C] -> [C,1,d] ✓
        # clip_weights 形状是 [C, d]
        self.clip_prototypes = clip_weights.T.unsqueeze(0)  # [1, d, C]

        # 本地特征/标签 buffer（每个 sync 窗口积累）
        self.feature_buffer = []  # list of [1, d] float32 CPU tensor
        self.label_buffer   = []  # list of int

        # 当前窗口内 zero-shot 临时 correct（synchronize 后用 StatA 结果替换）
        self._window_zs_correct = 0
        self._window_size       = 0

    @torch.no_grad()
    def evaluate_one(self):
        """读取一个样本，用 zero-shot 临时预测并缓存特征。"""
        if self.curr_idx >= self.num_samples:
            return 0, 0

        data, label = self.dataset[self.curr_idx]
        self.curr_idx += 1

        dataset = self.dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        is_cached = hasattr(dataset, 'is_cached_features') and dataset.is_cached_features

        # 加载/编码特征
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

        label_int = int(label.item()) if hasattr(label, 'item') else int(label)

        # Zero-shot 临时预测（实时计数用）
        zs_logits  = 100.0 * feat @ self.clip_weights.T  # [1, C]
        zs_pred    = int(zs_logits.argmax(dim=1).item())
        zs_correct = int(zs_pred == label_int)

        # 累积到总 correct（synchronize 后会用 StatA 结果修正）
        self.num_correct         += zs_correct
        self._window_zs_correct  += zs_correct
        self._window_size        += 1

        # 缓存特征（存 CPU float32，节省 GPU 内存）
        self.feature_buffer.append(feat.cpu().float())
        self.label_buffer.append(label_int)

        return zs_correct, 1

    def run_stata_on_buffer(self):
        """
        用本地 buffer 独立运行 StatA_solver，返回修正后的 correct 数。
        buffer 样本数 < 2 时（StatA 无法构建亲和矩阵），保留 zero-shot 结果。
        """
        n = len(self.feature_buffer)
        if n <= self.stata_n_neighbors + 1:
            return self._window_zs_correct

        # [N, d]
        query_features = torch.cat(self.feature_buffer, dim=0).float()
        # StatA_solver 不使用真实标签做监督，此处传全零占位
        query_labels = torch.zeros(n, dtype=torch.long)

        # 调用原版 StatA_solver（内部会将数据移到 cuda，返回 CPU tensor）
        _, z = StatA_solver(
            query_features,
            query_labels,
            self.clip_prototypes,           # [d, C, 1]
            alpha=self.stata_alpha,
            soft_beta=self.stata_soft_beta,
            lambda_y_hat=self.stata_lambda_y_hat,
            lambda_laplacian=self.stata_lambda_lap,
            n_neighbors=self.stata_n_neighbors,
            max_iter=self.stata_max_iter,
        )
        # z: [N, C] CPU，StatA 输出的软标签
        preds = z.argmax(dim=1).tolist()

        stata_correct = sum(
            int(preds[i] == self.label_buffer[i]) for i in range(n)
        )
        return stata_correct

    def flush_buffer(self):
        """清空当前窗口的 buffer 和临时计数。"""
        self.feature_buffer.clear()
        self.label_buffer.clear()
        self._window_zs_correct = 0
        self._window_size       = 0


# ==================== StatALocalServer ====================

class StatALocalServer(BaseCTTAServer):
    """
    StatA Local 基线服务器。

    synchronize() 时：
      - 对每个客户端独立调用 StatA_solver 处理本地 buffer
      - 用 StatA 结果替换该窗口内的 zero-shot 临时 correct
      - 客户端之间无任何通信（完全对齐论文 Local 版定义）
    """

    def __init__(self, datasets, clip_weights, args,
                 client_class=StatALocalClient):
        super().__init__(datasets, clip_weights, args, client_class)

    def syncronize(self):
        """
        Local 版同步：各客户端独立运行 StatA，无跨客户端通信。
        用 StatA 结果修正本窗口的 num_correct，然后清空 buffer。
        """
        for cid, client in self.clients.items():
            if not client.feature_buffer:
                continue

            stata_correct = client.run_stata_on_buffer()

            # 修正 num_correct：减去 zero-shot 临时值，加上 StatA 值
            client.num_correct -= client._window_zs_correct
            client.num_correct += stata_correct

            client.flush_buffer()


# ==================== Main ====================

def main():
    args = get_arguments()
    config_path = args.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the configured CLIP backbone.
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
        
        cfg = get_stata_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        args.config = cfg
        
        # Load the selected domain or corruption benchmark.
        domain_loaders, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess, separate_domains=True)
        clip_weights = clip_classifier(classnames, template, clip_model)
            
        if args.separate_domains:
            # ---- Separate-domain evaluation ----
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name} - Separate Domain Mode [StatA-Local]")
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
                
                # Partition this domain into client streams.
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

                server = StatALocalServer(client_datasets, clip_weights, args,
                                         client_class=StatALocalClient)
                acc, total_correct, total_samples, stats = server.evaluate()
                
                domain_results[domain_name] = {
                    'accuracy': acc, 'correct': total_correct, 'total': total_samples
                }
                total_correct_all += total_correct
                total_samples_all += total_samples
                print(f"  {domain_name}: {acc * 100:.2f}% ({total_correct}/{total_samples})")
            
            overall_acc = total_correct_all / total_samples_all
            print(f"\n{'='*60}")
            print(f"Results for {dataset_name} [StatA-Local]:")
            print(f"{'='*60}")
            for domain_name, res in domain_results.items():
                print(f"  {domain_name:20s}: {res['accuracy'] * 100:6.2f}%")
            print(f"  {'Total':20s}: {overall_acc * 100:6.2f}%"
                  f" ({total_correct_all}/{total_samples_all})")
            print(f"{'='*60}\n")
            
            if args.wandb:
                run_name = f"{dataset_name}_stata_local_separate"
                run = wandb.init(project="MCC-TTA-Baselines", config=cfg,
                                 group=group_name, name=run_name)
                for domain_name, res in domain_results.items():
                    wandb.log({f"{dataset_name}/{domain_name}": res['accuracy'] * 100})
                wandb.log({f"{dataset_name}/Total": overall_acc * 100})
                run.finish()
        
        else:
            # ---- Collaborative evaluation ----
            num_domains = len(domain_loaders)
            total_clients = num_domains * args.num_clients
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name} - Collaborative Mode [StatA-Local]")
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
                run_name = f"{dataset_name}_stata_local_collaborative"
                run = wandb.init(project="MCC-TTA-Baselines", config=cfg,
                                 group=group_name, name=run_name)

            server = StatALocalServer(all_client_datasets, clip_weights, args,
                                      client_class=StatALocalClient)
            acc, total_correct, total_samples, stats = server.evaluate()
            
            print(f"\n{'='*60}")
            print(f"Overall Results [StatA-Local]:")
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
