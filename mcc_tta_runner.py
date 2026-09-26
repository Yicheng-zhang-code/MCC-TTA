"""
MCC-TTA runner for federated test-time adaptation.
"""

import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from bisect import bisect_left
import math

import torch
import torch.nn.functional as F
import clip
import numpy as np
from torch.utils.data import Subset

"""
MCC-TTA: Mitigating Class Confusion for Federated Test-Time Adaptation
of Vision-Language Models.
Supported datasets: VLCS, TerraIncognita, CIFAR10CFull, CIFAR100CFull.
"""

from utils import *
from datasets.federated_loader import cache_features


# ==================== 参数解析 ====================

def get_arguments():
    """Get arguments of the federated test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, 
                        help='settings of MCC-TTA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', 
                        help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, 
                        help="Datasets to process, separated by a slash (/). Example: VLCS/TerraIncognita")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', 
                        help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, 
                        help='CLIP model backbone to use: RN50 or ViT-B/16.')
    
    # Federated learning specific parameters
    parser.add_argument('--num-clients', dest='num_clients', type=int, default=10,
                        help='Number of clients in federated setting.')
    parser.add_argument('--part-rate', dest='part_rate', type=float, default=1.0,
                        help='Participation rate of clients per round.')
    parser.add_argument('--sync-freq', dest='sync_freq', type=int, default=100,
                        help='Synchronization frequency (sync every N samples).')
    parser.add_argument('--partition', dest='partition', type=str, default='iid',
                        choices=['iid', 'domain', 'corruption'],
                        help='Data partitioning strategy: iid (random), domain (VLCS/Terra), corruption (CIFAR-C).')
    
    # Decentralized neighbor synchronization parameters
    parser.add_argument('--topology', dest='topology', type=str, default='similarity',
                        choices=['similarity', 'ring', 'random', 'full'],
                        help='Neighbor topology: similarity (centralized, default), ring, random, full (decentralized).')
    parser.add_argument('--neighbor-k', dest='neighbor_k', type=int, default=2,
                        help='Number of neighbors for decentralized topologies (ring/random).')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--cache-features', dest='cache_features', action='store_true',
                        help='Whether to pre-cache CLIP features for speedup.')
    parser.add_argument('--separate-domains', dest='separate_domains', action='store_true',
                        help='Evaluate each domain/corruption separately.')
    parser.add_argument('--cifar-protocol', dest='cifar_protocol', type=str, default='full',
                        choices=['full', 'nonoverlap'],
                        help=(
                            'Protocol for CIFAR-C collaborative mode. '
                            'full: each corruption stream is fully split into clients; '
                            'nonoverlap: original-image indices are assigned to only one corruption domain.'
                        ))
    parser.add_argument('--client_partition', dest='client_partition', type=str,
                        default='original', choices=['original', 'dirichlet'],
                        help='Client partition for rebuttal experiments. original keeps existing behavior.')
    parser.add_argument('--dirichlet_alpha', dest='dirichlet_alpha', type=float, default=0.5,
                        help='Dirichlet alpha metadata for fixed partition experiments.')
    parser.add_argument('--partition_seed', dest='partition_seed', type=int, default=2026,
                        help='Partition seed metadata for fixed partition experiments.')
    parser.add_argument('--partition_file', dest='partition_file', type=str, default=None,
                        help='Path to a fixed .npz client partition file.')
    parser.add_argument('--rebuttal_diagnostic', dest='rebuttal_diagnostic', action='store_true',
                        help='Record read-only direct mechanism diagnostics for rebuttal experiments.')
    parser.add_argument('--diagnostic-output-dir', dest='diagnostic_output_dir', type=str,
                        default='results/rebuttal',
                        help='Output directory for rebuttal diagnostic files.')
    parser.add_argument('--gaussian_label_noise', dest='gaussian_label_noise', type=float,
                        default=0.0,
                        help='Additional pseudo-label noise used only for Gaussian statistics updates.')
    parser.add_argument('--gaussian_noise_seed', dest='gaussian_noise_seed', type=int,
                        default=2026,
                        help='Independent seed for rebuttal Gaussian label-noise experiments.')
    parser.add_argument('--gaussian-noise-report', dest='gaussian_noise_report',
                        action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()
    return args


# ==================== Partition helpers ====================

def is_cifar_c_dataset(dataset_name):
    """Return True for CIFAR-10-C/100-C full corruption benchmarks."""
    return dataset_name in {"CIFAR10CFull", "CIFAR100CFull"}


def get_module_config(config, old_key, new_key):
    """Read MCC-TTA module configs using either legacy or paper-facing names."""
    return config.get(old_key, config.get(new_key, {}))


def setup_gaussian_noise_injector(args, num_classes):
    """Create the rebuttal-only Gaussian update noise hook when requested."""
    ratio = float(getattr(args, 'gaussian_label_noise', 0.0))
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("--gaussian_label_noise must be in [0, 1].")

    if ratio > 0.0 or getattr(args, 'gaussian_noise_report', False):
        from scripts.rebuttal_noise import GaussianLabelNoiseInjector
        args.gaussian_noise_injector = GaussianLabelNoiseInjector(
            num_classes=num_classes,
            noise_ratio=ratio,
            seed=getattr(args, 'gaussian_noise_seed', 2026),
        )
    else:
        args.gaussian_noise_injector = None


def print_gaussian_noise_report(args):
    """Print Gaussian label-noise counts for robustness experiments."""
    injector = getattr(args, 'gaussian_noise_injector', None)
    if injector is None or not getattr(args, 'gaussian_noise_report', False):
        return

    stats = injector.stats
    print(f"Gaussian label-noise updates: {stats.total_updates}")
    print(f"Artificially corrupted Gaussian updates: {stats.corrupted_updates}")
    print(f"Realized Gaussian label-noise ratio: {stats.realized_noise:.6f}")


def extract_labels_for_partition(dataset):
    """Extract labels from a raw or cached dataset without changing cache/config logic."""
    if hasattr(dataset, 'targets'):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        return np.asarray(labels, dtype=np.int64)
    if hasattr(dataset, 'tensors') and len(dataset.tensors) >= 2:
        labels = dataset.tensors[1]
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        return np.asarray(labels, dtype=np.int64)

    labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels.append(int(label.item()) if hasattr(label, 'item') else int(label))
    return np.asarray(labels, dtype=np.int64)


def stratified_nonoverlap_folds(labels, n_splits, seed=0):
    """
    Non-overlap control folds for corruption datasets.

    The same original-image index is assigned to exactly one fold among
    num_corruptions * clients_per_corruption folds. Each corruption-client
    later receives one fold. This prevents the same original image from
    appearing under multiple corruption types in the evaluated client set.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if n_splits <= 1:
        return [np.arange(len(labels))]

    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [fold for _, fold in skf.split(np.arange(len(labels)), labels)]
    except Exception:
        # Lightweight fallback: stratified round-robin split.
        rng = np.random.RandomState(seed)
        folds = [[] for _ in range(n_splits)]
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            for pos, sample_idx in enumerate(cls_idx):
                folds[pos % n_splits].append(int(sample_idx))
        folds = [np.asarray(fold, dtype=np.int64) for fold in folds]

    rng = np.random.RandomState(seed)
    for fold in folds:
        rng.shuffle(fold)

    # Keep the original corruption-fold construction deterministic.
    rng = np.random.RandomState(0)
    folds = list(folds)
    rng.shuffle(folds)
    return folds


# ==================== BaseClient 类（来自 Base.py）====================

class BaseClient:
    """基础客户端类"""

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
        
        # 从 args 中获取 clip_model
        self.clip_model = getattr(args, "clip_model", None)
        self.client_id = None
        self.diagnostic_logger = getattr(args, "diagnostic_logger", None)
        self._diagnostic_sample_uid = None

    @torch.no_grad()
    def predict(self, image_feature):
        """给定图像特征，返回预测结果"""
        # image_feature 必须已经是 CLIP 特征（已归一化）
        clip_logits, pred, _, _ = get_clip_logits(image_feature, self.clip_weights, normalize=False)
        return pred

    def evaluate_one(self):
        """评估单个样本"""
        if self.curr_idx >= self.num_samples:
            return 0, 0

        sample_idx = self.curr_idx
        data, label = self.dataset[self.curr_idx]
        self.curr_idx += 1

        diag_logger = self.diagnostic_logger if getattr(self.args, "rebuttal_diagnostic", False) else None
        if diag_logger is not None:
            original_index = sample_idx
            if hasattr(self.dataset, "indices"):
                original_index = int(self.dataset.indices[sample_idx])
            self._diagnostic_sample_uid = diag_logger.start_query(
                client_id=self.client_id,
                sample_index=sample_idx,
                original_index=original_index,
            )

        # 检查是否是缓存的特征（已归一化）
        # 需要处理 Subset 包装的情况
        dataset = self.dataset
        if hasattr(dataset, 'dataset'):  # Subset 有 dataset 属性指向底层数据集
            dataset = dataset.dataset
        is_cached = hasattr(dataset, 'is_cached_features') and dataset.is_cached_features
        
        # 判断是图像还是特征
        if data.dim() == 3 or (data.dim() == 4 and data.shape[0] == 1):
            # 原始图像 [C, H, W] 或 [1, C, H, W]，需要编码
            if data.dim() == 3:
                data = data.unsqueeze(0)  # [1, C, H, W]
            data = data.to(device=self.device)
            with torch.no_grad():
                image_feature = self.clip_model.encode_image(data)
                image_feature = image_feature.to(dtype=self.dtype)
                # 编码后需要归一化
                image_feature = F.normalize(image_feature, dim=1)
        else:
            # 已经是 CLIP 特征 [feat_dim] 或 [1, feat_dim]
            image_feature = data.to(device=self.device, dtype=self.dtype)
            
            # 确保是 [1, feat_dim] 的形状
            if image_feature.dim() == 1:
                image_feature = image_feature.unsqueeze(0)
            
            # 只有非缓存特征才需要归一化（缓存特征已经归一化过了）
            if not is_cached:
                image_feature = F.normalize(image_feature, dim=1)

        pred = self.predict(image_feature)

        # label 可能是 Python int 或 torch.Tensor
        label_int = int(label.item()) if hasattr(label, "item") else int(label)
        correct = int(pred == label_int)
        self.num_correct += correct

        if diag_logger is not None:
            diag_logger.finish_query(
                sample_uid=self._diagnostic_sample_uid,
                ground_truth=label_int,
                final_pred=pred,
                correct=correct,
            )
            self._diagnostic_sample_uid = None

        return correct, 1

    def is_done(self):
        """检查是否完成所有样本"""
        return self.curr_idx >= self.num_samples


# ==================== BaseCTTAServer 类（来自 Base.py）====================

class BaseCTTAServer:
    """协作测试时适应服务器基类"""

    def __init__(self, datasets, clip_weights, args, client_class=BaseClient):
        self.num_class, self.feat_dim = clip_weights.shape

        self.clients = OrderedDict(
            [(cid, client_class(dataset, clip_weights, args)) for (cid, dataset) in datasets.items()])
        for cid, client in self.clients.items():
            client.client_id = cid
            client.diagnostic_logger = getattr(args, "diagnostic_logger", None)

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
        """评估所有客户端（支持协作）"""
        total_correct, total_num_samples = 0, 0

        for rnd in tqdm(range(1, self.num_rounds + 1), desc="Federated TTA"):
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

        stats = {cid: client.num_correct / client.num_samples for (cid, client) in self.clients.items()}

        return total_correct / total_num_samples, total_correct, total_num_samples, stats

    def syncronize(self):
        """同步方法（子类重载）"""
        pass


# ==================== MCC-TTA Client ====================

class MccTtaClient(BaseClient):
    """MCC-TTA client with local and externally shared memory."""

    def __init__(self, dataset, clip_weights, args):
        super(MccTtaClient, self).__init__(dataset, clip_weights, args)

        self.config = args.config
        # 从 args 中获取 clip_model（在 main() 中设置）
        self.clip_model = getattr(args, "clip_model", None)

        local_cfg, global_cfg = args.config['local'], args.config['global']

        self.local_enabled, self.global_enabled = local_cfg['enabled'], global_cfg['enabled']

        self.local_params = {k: local_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'gamma']}
        self.global_params = {k: global_cfg[k] for k in ['shot_capacity', 'prototype', 'gamma']}

        # Note: actual num_clients will be set by server after all clients are created
        # This is just a placeholder, will be updated by server
        self.global_params['shot_capacity'] = self.global_params['shot_capacity']

        # 本地内存
        self.local_ent = [[] for _ in range(self.num_class)]  # 单维度优先级队列

        self.local_cache = torch.zeros(self.num_class, self.local_params['shot_capacity'], self.feat_dim,
                                       device=self.device, dtype=self.dtype)

        self.local_cache_ent = torch.ones(self.num_class, self.local_params['shot_capacity'],
                                          device=self.device, dtype=self.dtype)

        # 外部内存
        self.external_cache = torch.zeros(self.num_class, self.global_params['shot_capacity'], self.feat_dim,
                                          device=self.device, dtype=self.dtype)

        self.external_cache_ent = torch.ones(self.num_class, self.global_params['shot_capacity'],
                                             device=self.device, dtype=self.dtype)

        self.max_ent = math.log(self.num_class)

        # === 新增：低秩模块配置（双维度：熵 + 文本重构误差）===
        self.lr_config = get_module_config(args.config, 'low_rank', 'memory_reg')
        self.lr_enabled = self.lr_config.get('enabled', False)
        
        if self.lr_enabled and self.lr_config.get('use_recon_error', False):
            # 文本重构误差缓存
            self.local_cache_recon = torch.ones(
                self.num_class, self.local_params['shot_capacity'],
                device=self.device, dtype=self.dtype)
            self.external_cache_recon = torch.ones(
                self.num_class, self.global_params['shot_capacity'],
                device=self.device, dtype=self.dtype)
            
            # 双维度权重
            self.alpha_H = self.lr_config.get('alpha_H', 0.2)
            self.alpha_R_text = self.lr_config.get('alpha_R_text', 0.8)
            
            # 双维度优先级队列（独立于单维度的 local_ent）
            self.local_score_lists = [[] for _ in range(self.num_class)]

        # === Corruption-Aware Logit Calibration (CALC) 配置 ===
        calc_cfg = get_module_config(args.config, 'corruption_calibration', 'gauss_calib')
        self.calc_enabled = calc_cfg.get('enabled', False)
        self.calc_alpha   = calc_cfg.get('alpha', 0.5)
        # Welford 在线统计量（float32，GPU）：用所有历史样本估计每类受污染分布
        self.calc_mu  = torch.zeros(self.num_class, self.feat_dim, device=self.device, dtype=torch.float32)  # 在线均值
        self.calc_M2  = torch.zeros(self.num_class, self.feat_dim, device=self.device, dtype=torch.float32)  # 二阶矩累积
        self.calc_n   = torch.zeros(self.num_class, device=self.device, dtype=torch.float32)                 # 每类样本数
        self.calc_var = torch.ones(self.num_class, self.feat_dim, device=self.device, dtype=torch.float32)   # 方差（从M2算）
        self._calc_all_ready = False  # 所有类至少有1个样本时置True
        self._calc_class_filled = torch.zeros(self.num_class, dtype=torch.bool)  # 各类是否有样本
        # sample_threshold：进入多少个样本后开启 CALC（0 表示使用旧的「所有类有样本」逻辑）
        self._calc_threshold = int(calc_cfg.get('sample_threshold', 0))
        self._calc_sample_count = 0  # 累计进入的样本数，用于 sample_threshold 模式
        # 细粒度判别校准配置
        self._fg_config = get_module_config(args.config, 'fine_grain', 'score_corr')
        # 预计算上三角掩码（固定值，缓存避免每次 predict 重新分配）
        self._fg_triu = torch.triu(
            torch.ones(self.num_class, self.num_class, dtype=torch.bool, device=self.device),
            diagonal=1)
        # 缓存 get_clip_logits 签名检查结果，避免每次 predict 调用 inspect
        import inspect as _inspect
        _params = list(_inspect.signature(get_clip_logits).parameters.keys())
        self._clip_logits_legacy = (len(_params) >= 2 and _params[1] == 'clip_model')

    @torch.no_grad()
    def _compute_reconstruction_error(self, feature, y_pred):
        """计算文本重构误差
        
        Args:
            feature: 图像特征 [d]
            y_pred: 预测类别
        
        Returns:
            R_text: 文本重构误差，归一化到 [0, 1]
        """
        if not self.lr_enabled or not self.lr_config.get('use_recon_error', False):
            return 0.0

        text_emb = self.clip_weights[y_pred]  # [d]
        cosine_sim = (feature @ text_emb).item()
        R_text = (1.0 - cosine_sim) / 2.0  # 归一化到 [0, 1]
        
        return R_text

    @torch.no_grad()
    def update_cache(self, cache, cache_ent, ent_lists, pred, features_loss, shot_capacity, cache_recon=None):
        """更新本地缓存（支持单维度和双维度）
        
        Args:
            cache: 缓存特征
            cache_ent: 缓存熵
            ent_lists: 单维度优先级队列
            pred: 预测类别
            features_loss: [feature, entropy]
            shot_capacity: 容量
            cache_recon: 文本重构误差缓存 (可选，启用双维度)
        """
        feature, new_ent = features_loss[:2]
        
        # 确保 feature 是 [d] 而不是 [1, d]
        if feature.dim() == 2 and feature.shape[0] == 1:
            feature = feature.squeeze(0)

        # 计算重构误差和综合分数
        if cache_recon is not None:
            # 双维度模式
            new_recon_text = self._compute_reconstruction_error(feature, pred)
            new_ent_normalized = new_ent / self.max_ent
            new_score = self.alpha_H * new_ent_normalized + self.alpha_R_text * new_recon_text
            score_lists = self.local_score_lists  # 使用双维度优先级队列
        else:
            # 单维度模式
            new_recon_text = 0.0
            new_score = new_ent / self.max_ent  # 保持与双维度一致的归一化
            score_lists = ent_lists  # 使用单维度优先级队列

        retained = False
        retained_slot = None

        # 队列更新逻辑
        if len(score_lists[pred]) >= shot_capacity:
            # 队列已满，从末尾取最差样本
            worst_score, worst_idx = score_lists[pred][-1]

            # 如果新样本更好，替换
            if new_score < worst_score:
                retained = True
                retained_slot = worst_idx
                cache[pred][worst_idx] = feature
                cache_ent[pred][worst_idx] = new_ent / self.max_ent
                if cache_recon is not None:
                    cache_recon[pred][worst_idx] = new_recon_text
                
                # 更新优先级队列
                score_lists[pred].pop()
                new_item = (new_score, worst_idx)
                insert_idx = bisect_left(score_lists[pred], new_item)
                score_lists[pred].insert(insert_idx, new_item)
        else:
            # 队列未满，直接插入
            idx = len(score_lists[pred])
            retained = True
            retained_slot = idx
            cache[pred][idx] = feature
            cache_ent[pred][idx] = new_ent / self.max_ent
            if cache_recon is not None:
                cache_recon[pred][idx] = new_recon_text
            
            # 更新优先级队列
            new_item = (new_score, idx)
            insert_idx = bisect_left(score_lists[pred], new_item)
            score_lists[pred].insert(insert_idx, new_item)

        diag_logger = self.diagnostic_logger if getattr(self.args, "rebuttal_diagnostic", False) else None
        if diag_logger is not None:
            diag_logger.record_memory_event(
                sample_uid=self._diagnostic_sample_uid,
                client_id=self.client_id,
                pseudo_label=pred,
                candidate=True,
                retained=retained,
                class_index=pred,
                slot_index=retained_slot,
                entropy=float(new_ent),
                normalized_entropy=float(new_ent / self.max_ent),
                reconstruction_error=float(new_recon_text),
                regulation_score=float(new_score),
                dual_criteria=cache_recon is not None,
            )

        # CALC Welford 在线更新：每个样本都参与，不受缓存门槛限制
        # 3次 [d] 元素运算，极快，无额外内存分配
        if self.calc_enabled:
            f_w = feature.float()                         # [d] float32，避免fp16精度损失
            gaussian_update_label = pred
            noise_injector = getattr(self.args, 'gaussian_noise_injector', None)
            if noise_injector is not None:
                gaussian_update_label = noise_injector.maybe_corrupt(
                    pred, client_id=self.client_id)

            self.calc_n[gaussian_update_label] += 1.0
            delta  = f_w - self.calc_mu[gaussian_update_label]             # [d]
            self.calc_mu[gaussian_update_label].add_(delta / self.calc_n[gaussian_update_label])  # 原地更新均值
            delta2 = f_w - self.calc_mu[gaussian_update_label]             # [d]
            self.calc_M2[gaussian_update_label].add_(delta * delta2)       # 原地更新二阶矩
            if self.calc_n[gaussian_update_label] > 1:
                self.calc_var[gaussian_update_label] = (
                    self.calc_M2[gaussian_update_label] / self.calc_n[gaussian_update_label]
                ).clamp(min=1e-4)
            # 样本计数器：累计样本数达到阈值后启用CALC
            if not self._calc_all_ready:
                self._calc_sample_count += 1
                if self._calc_threshold > 0:
                    # sample_threshold 模式：累计样本数达标即开启
                    self._calc_all_ready = (self._calc_sample_count >= self._calc_threshold)
                else:
                    # 兼容旧逻辑：所有类都至少有1个样本才开启
                    self._calc_class_filled[gaussian_update_label] = True
                    self._calc_all_ready = bool(self._calc_class_filled.all())

    @torch.no_grad()
    def upload_cache(self, prototype='exp_weighted', gamma=1.0):
        """上传原型到服务器"""
        if prototype == 'mean':
            prototype_to_upload = F.normalize(self.local_cache.mean(dim=1), dim=1)
            ents = get_entropy_batch(prototype_to_upload, self.clip_weights) / self.max_ent
            recons = self.local_cache_recon.mean(dim=1) if self.lr_enabled and self.lr_config.get('use_recon_error', False) else None

        elif prototype == 'exp_weighted':
            # 双维度加权：启用文本重构误差时用综合分数，否则退化为纯熵
            if self.lr_enabled and self.lr_config.get('use_recon_error', False):
                combined_scores = self.alpha_H * self.local_cache_ent + self.alpha_R_text * self.local_cache_recon
                weights = (- gamma * combined_scores).exp().unsqueeze(2)
            else:
                weights = (- gamma * self.local_cache_ent).exp().unsqueeze(2)
            proto_raw = (self.local_cache * weights).mean(dim=1)  # [C, d] 未归一化

            prototype_to_upload = F.normalize(proto_raw, dim=1)
            ents = get_entropy_batch(prototype_to_upload, self.clip_weights) / self.max_ent
            
            # 加权平均文本重构误差
            if self.lr_enabled and self.lr_config.get('use_recon_error', False):
                weights_1d = weights.squeeze(2)
                recons = (self.local_cache_recon * weights_1d).sum(dim=1) / weights_1d.sum(dim=1)
            else:
                recons = None

        elif prototype == 'min_ent':
            # 根据是否启用双维度选择正确的优先级队列
            if self.lr_enabled and self.lr_config.get('use_recon_error', False):
                score_lists = self.local_score_lists  # 双维度队列
            else:
                score_lists = self.local_ent  # 单维度队列
            
            min_ent_idx = torch.LongTensor(
                [score_list[0][1] if score_list else 0 for score_list in score_lists])

            class_idx = torch.arange(self.num_class)
            prototype_to_upload = self.local_cache[class_idx, min_ent_idx]
            ents = self.local_cache_ent[class_idx, min_ent_idx]
            recons = self.local_cache_recon[class_idx, min_ent_idx] if self.lr_enabled and self.lr_config.get('use_recon_error', False) else None
        else:
            raise NotImplementedError(f"Unknown prototype method: {prototype}")

        return prototype_to_upload, ents, recons

    @torch.no_grad()
    def download_cache(self, external_cache, external_cache_ent, external_cache_recon=None):
        """从服务器下载外部内存"""
        self.external_cache = external_cache
        self.external_cache_ent = external_cache_ent
        if external_cache_recon is not None:
            self.external_cache_recon = external_cache_recon

    @torch.no_grad()
    def _corruption_calibrate(self, image_feature):
        """Corruption-Aware Logit Calibration

        用本地缓存统计的每类受污染分布（对角高斯）计算当前样本的对数概率，
        作为校准项叠加到 final_logits 上。

        Returns:
            calib_logits: [1, C] float32
        """
        f   = image_feature.float()   # [1, d]
        mu  = self.calc_mu          # [C, d]，Welford在线均值，已在GPU float32
        var = self.calc_var         # [C, d]，Welford在线方差，已在GPU float32

        diff     = f - mu                                             # [C, d]
        log_prob = -0.5 * ((diff ** 2 / var) + var.log()).sum(dim=1)  # [C]

        return log_prob.unsqueeze(0)  # [1, C]

    def compute_cache_logits(self, image_feature, cache, cache_ent, alpha, beta, gamma):
        """计算内存增强的logits（论文原始实现）
        
        Args:
            image_feature: [1, feat_dim]
            cache: [num_class, shot_capacity, feat_dim]
            cache_ent: [num_class, shot_capacity]
            alpha, beta, gamma: 超参数
        """
        affinity = cache @ image_feature.T  # [num_class, shot_capacity, batch_size]
        attn = (beta * (affinity - 1)).exp()

        attn.mul_((- gamma * cache_ent.unsqueeze(2)).exp())



        adaptive_cls_weight = (cache.unsqueeze(3) * attn.unsqueeze(2)).sum(dim=1)
        adaptive_cls_weight = adaptive_cls_weight.permute(2, 0, 1)

        # 论文原始逻辑：通过 norm 检查空向量（未填充位置自然为零向量）
        is_empty = adaptive_cls_weight.norm(dim=2) <= 1e-3
        adaptive_cls_weight = F.normalize(adaptive_cls_weight, dim=2)
        adaptive_cls_weight[is_empty] = 0.0

        cache_logits = 100.0 * (image_feature.unsqueeze(1) * adaptive_cls_weight).sum(dim=2)

        return alpha * cache_logits

    @torch.no_grad()
    def predict(self, image_feature):
        """预测（带内存增强）"""
        diag_logger = self.diagnostic_logger if getattr(self.args, "rebuttal_diagnostic", False) else None

        # 1. 获取CLIP预测（签名在 __init__ 已缓存，无 inspect 开销）
        if self._clip_logits_legacy:
            clip_logits, pred, proba, entropy, image_feature = get_clip_logits(
                image_feature, None, self.clip_weights, return_features=True)
        else:
            clip_logits, pred, proba, entropy, image_feature = get_clip_logits(
                image_feature, self.clip_weights, return_features=True)

        # 2. 更新本地内存（双维度版本：熵 + 文本重构误差）
        if self.lr_enabled and self.lr_config.get('use_recon_error', False):
            self.update_cache(
                self.local_cache, self.local_cache_ent, self.local_ent,
                pred, [image_feature, entropy], self.local_params['shot_capacity'],
                cache_recon=self.local_cache_recon)
        else:
            # 原版更新（只用熵）
            self.update_cache(
                self.local_cache, self.local_cache_ent, self.local_ent,
                pred, [image_feature, entropy], self.local_params['shot_capacity'])

        local_memory_counts = None
        if diag_logger is not None:
            local_memory_counts = diag_logger.get_local_coverage_counts(self.client_id, self.num_class)

        # 3. 融合本地和外部内存
        final_logits = clip_logits.clone()

        if self.global_enabled and self.local_enabled:
            cache = torch.cat([self.local_cache, self.external_cache], dim=1)
            cache_ent = torch.cat([self.local_cache_ent, self.external_cache_ent], dim=1)

            # 双维度筛选：熵 + 文本重构误差
            if self.lr_enabled and self.lr_config.get('use_recon_error', False):
                # 确保外部缓存也有重构误差（应该由服务器保证）
                if not hasattr(self, 'external_cache_recon'):
                    self.external_cache_recon = torch.ones(
                        self.num_class, self.external_cache.shape[1],
                        device=self.device, dtype=self.dtype)
                
                cache_recon = torch.cat([self.local_cache_recon, self.external_cache_recon], dim=1)
                cache_scores = self.alpha_H * cache_ent + self.alpha_R_text * cache_recon
                selected = cache_scores.argsort(dim=1, descending=False)[:, :self.local_params['shot_capacity']]
            else:
                # 单维度：只用熵
                selected = cache_ent.argsort(dim=1, descending=False)[:, :self.local_params['shot_capacity']]
            
            cache = torch.gather(cache, 1, selected.unsqueeze(-1).expand(-1, -1, self.feat_dim))
            cache_ent = torch.gather(cache_ent, 1, selected)

        elif self.local_enabled:
            cache = self.local_cache
            cache_ent = self.local_cache_ent

        elif self.global_enabled:
            cache = self.external_cache
            cache_ent = self.external_cache_ent
        else:
            return final_logits.argmax(dim=1).item()

        # 4. 计算内存增强
        cache_logits_delta = self.compute_cache_logits(image_feature, cache, cache_ent,
                                                  self.local_params['alpha'],
                                                  self.local_params['beta'],
                                                  self.local_params['gamma'])
        final_logits += cache_logits_delta

        score_before_gaussian = None
        score_after_gaussian = None
        score_before_pairwise = None
        score_after_pairwise = None
        activated_pairs = []
        gaussian_applied = False

        if diag_logger is not None:
            score_before_gaussian = final_logits.detach().float().cpu().squeeze(0).clone()

        # 5. Corruption-Aware Logit Calibration
        if self.calc_enabled and self._calc_all_ready:
            calib = self._corruption_calibrate(image_feature)
            final_logits = final_logits.float() + self.calc_alpha * calib
            gaussian_applied = True

        if diag_logger is not None:
            score_after_gaussian = final_logits.detach().float().cpu().squeeze(0).clone()
            score_before_pairwise = score_after_gaussian.clone()

        # 6. 细粒度判别校准（Proto-Disc）
        # 参数：similarity_threshold, lambda, proto_alpha
        # proto_alpha=1: 纯视觉原型；proto_alpha=0: 纯CLIP文本权重；中间值为融合
        fg_cfg = self._fg_config
        if fg_cfg.get('enabled', False):
            fg_lambda     = fg_cfg.get('lambda', 1.0)
            fg_thresh     = fg_cfg.get('similarity_threshold', 0.85)
            fg_alpha      = float(fg_cfg.get('proto_alpha', 1.0))  # 视觉权重

            # 融合视觉原型与CLIP文本权重
            vision_proto = cache.mean(dim=1).float()           # [C, d]
            text_proto   = self.clip_weights.float()           # [C, d] 已归一化
            if fg_alpha >= 1.0:
                proto = vision_proto
            elif fg_alpha <= 0.0:
                proto = text_proto                             # 直接用归一化文本权重
            else:
                # 文本原型缩放到与视觉原型相同量级再融合
                v_norm = vision_proto.norm(dim=1, keepdim=True).mean().clamp(min=1e-6)
                text_scaled = F.normalize(text_proto, dim=1) * v_norm
                proto = fg_alpha * vision_proto + (1.0 - fg_alpha) * text_scaled

            valid = proto.norm(dim=1) > 1e-6
            proto_norm = F.normalize(proto, dim=1)
            sim = (proto_norm @ proto_norm.T)                   # [C, C]
            pair_mask = self._fg_triu & (sim > fg_thresh) \
                             & valid.unsqueeze(1) & valid.unsqueeze(0)
            a_idx, b_idx = pair_mask.nonzero(as_tuple=True)
            if diag_logger is not None:
                activated_pairs = [
                    (int(a), int(b))
                    for a, b in zip(a_idx.detach().cpu().tolist(), b_idx.detach().cpu().tolist())
                ]
            if a_idx.numel() > 0:
                # 差向量校准：按sqrt(对数)缩放，统一适应不同类数数据集
                diff = proto[a_idx] - proto[b_idx]             # [P, d]
                pair_sims_val = sim[a_idx, b_idx].clamp(max=0.9999)
                hard_weight = 1.0 / (1.0 - pair_sims_val + 1e-4)
                hard_weight = hard_weight / hard_weight.sum()  # 全局归一化
                proj = (image_feature.float() @ diff.T).squeeze(0)
                num_pairs = float(a_idx.shape[0])
                scale = num_pairs ** 0.5                        # sqrt(P) 补偿多对场景
                delta = fg_lambda * scale * hard_weight * proj


                fl = final_logits.float()
                fl.scatter_add_(1, a_idx.unsqueeze(0),  delta.unsqueeze(0))
                fl.scatter_add_(1, b_idx.unsqueeze(0), -delta.unsqueeze(0))
                final_logits = fl

        final_pred = final_logits.argmax(dim=1).item()

        if diag_logger is not None:
            score_after_pairwise = final_logits.detach().float().cpu().squeeze(0).clone()
            diag_logger.record_query_scores(
                sample_uid=self._diagnostic_sample_uid,
                client_id=self.client_id,
                clip_pred=pred,
                clip_entropy=float(entropy),
                local_memory_counts=local_memory_counts,
                gaussian_applied=gaussian_applied,
                score_before_gaussian=score_before_gaussian,
                score_after_gaussian=score_after_gaussian,
                score_before_pairwise=score_before_pairwise,
                score_after_pairwise=score_after_pairwise,
                activated_pairs=activated_pairs,
            )

        return final_pred


# ==================== MCC-TTA Server ====================

class MccTtaServer(BaseCTTAServer):
    """MCC-TTA server that maintains and distributes shared memory."""

    def __init__(self, datasets, clip_weights, args, client_class=MccTtaClient):
        super(MccTtaServer, self).__init__(datasets, clip_weights, args, client_class)

        global_cfg = args.config['global']

        self.global_params = {k: global_cfg[k] for k in ['shot_capacity', 'prototype', 'gamma']}
        # Use actual number of clients (self.num_clients), not args.num_clients
        self.global_params['shot_capacity'] = min(self.global_params['shot_capacity'], self.num_clients - 1)

        # Update all clients' global_params with correct shot_capacity
        for client in self.clients.values():
            client.global_params['shot_capacity'] = self.global_params['shot_capacity']
            # Resize external cache if needed
            if client.external_cache.shape[1] != self.global_params['shot_capacity']:
                client.external_cache = torch.zeros(self.num_class, self.global_params['shot_capacity'], self.feat_dim,
                                                    device=self.device, dtype=self.dtype)
                client.external_cache_ent = torch.ones(self.num_class, self.global_params['shot_capacity'],
                                                       device=self.device, dtype=self.dtype)
                # 如果启用了文本重构误差，也需要调整大小
                if client.lr_enabled and client.lr_config.get('use_recon_error', False):
                    client.external_cache_recon = torch.ones(self.num_class, self.global_params['shot_capacity'],
                                                            device=self.device, dtype=self.dtype)

        self.global_cache = torch.zeros(self.num_class, self.num_clients, self.feat_dim,
                                        device=self.device, dtype=self.dtype)

        self.global_cache_ent = torch.ones(self.num_class, self.num_clients,
                                           device=self.device, dtype=self.dtype)
        
        # 全局文本重构误差缓存（如果任一客户端启用）
        self.lr_enabled = any(client.lr_enabled and client.lr_config.get('use_recon_error', False) 
                             for client in self.clients.values())
        if self.lr_enabled:
            self.global_cache_recon = torch.ones(self.num_class, self.num_clients,
                                                device=self.device, dtype=self.dtype)


        # 构建邻居拓扑（支持去中心化）
        self.topology_type = getattr(args, 'topology', 'similarity')
        self.neighbor_k = getattr(args, 'neighbor_k', 2)
        if self.topology_type != 'similarity':
            # 去中心化模式：构建邻居拓扑
            self.neighbor_topology = self._build_neighbor_topology(
                topology=self.topology_type, k=self.neighbor_k)
            if self.topology_type == 'random':
                print(f"Using decentralized topology: {self.topology_type} with k={self.neighbor_k} neighbors (dynamic, updates every sync)")
            else:
                print(f"Using decentralized topology: {self.topology_type} with k={self.neighbor_k} neighbors (static)")
        else:
            # 中心化模式：基于相似度选择
            self.neighbor_topology = None
            print("Using centralized similarity-based selection")

    def syncronize(self):
        """同步：收集原型并分发个性化外部内存"""
        selected_client_idxs = sorted(list(torch.randperm(self.num_clients)[:self.cohort_size].numpy()))

        # 动态拓扑：random 模式下每次同步时重新构建
        if self.topology_type == 'random':
            self.neighbor_topology = self._build_neighbor_topology(
                topology='random', k=self.neighbor_k)

        # 1. 收集原型
        for client_idx in selected_client_idxs:
            cid = self.idx2cid[client_idx]
            client = self.clients[cid]
            prototypes, ents, recons = client.upload_cache(prototype=self.global_params['prototype'],
                                                          gamma=self.global_params['gamma'])

            self.global_cache[:, client_idx, :] = prototypes
            self.global_cache_ent[:, client_idx] = ents
            if self.lr_enabled and recons is not None:
                self.global_cache_recon[:, client_idx] = recons

        # 2. 分发个性化外部内存
        for client_idx in selected_client_idxs:
            cid = self.idx2cid[client_idx]
            client = self.clients[cid]

            if self.lr_enabled and client.lr_enabled and client.lr_config.get('use_recon_error', False):
                # 传入客户端的双维度权重
                personal_cache, personal_cache_ent, personal_cache_recon = self.subset_selection(
                    self.global_cache, self.global_cache_ent, client_idx, self.global_params['shot_capacity'],
                    global_cache_recon=self.global_cache_recon,
                    alpha_H=client.alpha_H, alpha_R_text=client.alpha_R_text)
                client.download_cache(personal_cache, personal_cache_ent, personal_cache_recon)
            else:
                personal_cache, personal_cache_ent = self.subset_selection(
                    self.global_cache, self.global_cache_ent, client_idx, self.global_params['shot_capacity'])
                client.download_cache(personal_cache, personal_cache_ent)


    def _build_neighbor_topology(self, topology='ring', k=2):
        """构建邻居拓扑
        
        Args:
            topology: 'ring' (环形), 'random' (随机), 'full' (全连接)
            k: 邻居数量
        
        Returns:
            neighbors: dict, {client_idx: [neighbor_idx1, neighbor_idx2, ...]}
        """
        neighbors = {}
        
        if topology == 'ring':
            # 环形拓扑：每个节点连接前后 k//2 个邻居
            for i in range(self.num_clients):
                neighbors[i] = []
                for offset in range(1, k//2 + 1):
                    neighbors[i].append((i - offset) % self.num_clients)
                    neighbors[i].append((i + offset) % self.num_clients)
                neighbors[i] = list(set(neighbors[i]))  # 去重
        
        elif topology == 'random':
            # 随机拓扑：每个节点随机连接 k 个邻居
            for i in range(self.num_clients):
                candidates = list(range(self.num_clients))
                candidates.remove(i)
                neighbors[i] = random.sample(candidates, min(k, len(candidates)))
        
        elif topology == 'full':
            # 全连接：每个节点连接所有其他节点
            for i in range(self.num_clients):
                neighbors[i] = [j for j in range(self.num_clients) if j != i]
        
        return neighbors

    def subset_selection(self, global_cache, global_cache_ent, query_idx, topk, global_cache_recon=None, alpha_H=None, alpha_R_text=None):
        """为每个类别选择原型（支持中心化和去中心化）
        
        Args:
            global_cache: 全局缓存 [C, num_clients, d]
            global_cache_ent: 全局熵 [C, num_clients]
            query_idx: 查询客户端索引
            topk: 选择数量
            global_cache_recon: 全局重构误差 [C, num_clients] (可选)
            alpha_H: 熵权重 (可选，用于双维度选择)
            alpha_R_text: 文本重构误差权重 (可选，用于双维度选择)
        """
        
        if self.neighbor_topology is None:
            # 中心化模式：基于相似度选择最相似的topk个原型
            query_cache = global_cache[:, query_idx, :]
            similarity = (global_cache * query_cache.unsqueeze(1)).sum(dim=2)
            selected_idx = torch.sort(similarity, dim=1, descending=True).indices[:, 1:(topk + 1)]
            
            personal_cache_ent = global_cache_ent.gather(1, selected_idx)
            selected = selected_idx.unsqueeze(-1).expand(-1, -1, global_cache.shape[2])
            personal_cache = global_cache.gather(1, selected)
            
            if global_cache_recon is not None:
                personal_cache_recon = global_cache_recon.gather(1, selected_idx)
                return personal_cache, personal_cache_ent, personal_cache_recon
            else:
                return personal_cache, personal_cache_ent
        
        else:
            # 去中心化模式：只从邻居中选择原型
            neighbor_indices = self.neighbor_topology[query_idx]
            
            if len(neighbor_indices) == 0:
                # 没有邻居，返回空
                personal_cache = torch.zeros(self.num_class, topk, self.feat_dim,
                                            device=self.device, dtype=self.dtype)
                personal_cache_ent = torch.ones(self.num_class, topk,
                                               device=self.device, dtype=self.dtype)
                if global_cache_recon is not None:
                    personal_cache_recon = torch.ones(self.num_class, topk,
                                                     device=self.device, dtype=self.dtype)
                    return personal_cache, personal_cache_ent, personal_cache_recon
                else:
                    return personal_cache, personal_cache_ent
            
            # 提取邻居的原型
            neighbor_cache = global_cache[:, neighbor_indices, :]      # [num_class, num_neighbors, feat_dim]
            neighbor_ents = global_cache_ent[:, neighbor_indices]      # [num_class, num_neighbors]
            
            # 如果邻居数 > topk，选择最优的
            if len(neighbor_indices) > topk:
                # 双维度模式：按综合分数选择；单维度模式：按熵选择
                if global_cache_recon is not None and alpha_H is not None and alpha_R_text is not None:
                    neighbor_recons = global_cache_recon[:, neighbor_indices]
                    neighbor_scores = alpha_H * neighbor_ents + alpha_R_text * neighbor_recons
                    selected_local_idx = neighbor_scores.argsort(dim=1)[:, :topk]
                    personal_cache_recon = torch.gather(neighbor_recons, 1, selected_local_idx)
                else:
                    selected_local_idx = neighbor_ents.argsort(dim=1)[:, :topk]
                
                personal_cache = torch.gather(neighbor_cache, 1, 
                                             selected_local_idx.unsqueeze(-1).expand(-1, -1, self.feat_dim))
                personal_cache_ent = torch.gather(neighbor_ents, 1, selected_local_idx)
                

            else:
                # 邻居数 <= topk，全部使用，不足部分填充
                personal_cache = torch.zeros(self.num_class, topk, self.feat_dim,
                                            device=self.device, dtype=self.dtype)
                personal_cache_ent = torch.ones(self.num_class, topk,
                                               device=self.device, dtype=self.dtype)
                personal_cache[:, :len(neighbor_indices), :] = neighbor_cache
                personal_cache_ent[:, :len(neighbor_indices)] = neighbor_ents
                
                if global_cache_recon is not None:
                    neighbor_recons = global_cache_recon[:, neighbor_indices]
                    personal_cache_recon = torch.ones(self.num_class, topk,
                                                     device=self.device, dtype=self.dtype)
                    personal_cache_recon[:, :len(neighbor_indices)] = neighbor_recons
            
            if global_cache_recon is not None:
                return personal_cache, personal_cache_ent, personal_cache_recon
            else:
                return personal_cache, personal_cache_ent


# ==================== 主函数（参考 tda_runner.py）====================

def main():
    args = get_arguments()
    config_path = args.config
    args.diagnostic_logger = None
    args.gaussian_noise_injector = None

    if getattr(args, "rebuttal_diagnostic", False):
        from scripts.diagnose_direct_mechanisms import DirectMechanismDiagnosticLogger

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Initialize CLIP model
    result = clip.load(args.backbone, device=device)
    if len(result) == 3:
        # clip2 returns (model, embed_dim, preprocess)
        clip_model, embed_dim, preprocess = result
    else:
        # clip returns (model, preprocess)
        clip_model, preprocess = result
    clip_model.eval()
    # Attach clip_model to args so clients can encode images when --cache-features is OFF.
    args.clip_model = clip_model
    args.device = device

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"
    
    # Run MCC-TTA on each dataset.
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        # 将配置添加到 args
        args.config = cfg
        if getattr(args, "rebuttal_diagnostic", False):
            args.diagnostic_logger = DirectMechanismDiagnosticLogger(args.diagnostic_output_dir)
        else:
            args.diagnostic_logger = None
        
        # Load dataset
        if args.separate_domains:
            # 分别评估每个子数据集（论文中的方式）
            domain_loaders, classnames, template = build_test_data_loader(
                dataset_name, args.data_root, preprocess, separate_domains=True)
            clip_weights = clip_classifier(classnames, template, clip_model)
            setup_gaussian_noise_injector(args, clip_weights.shape[0])
            
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name} - Separate Domain Mode")
            print(f"{'='*60}\n")
            
            domain_results = {}
            total_correct_all = 0
            total_samples_all = 0
            
            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f"\n--- Evaluating domain: {domain_name} ---")
                
                dataset = domain_loader.dataset
                
                # Optional: cache features for speedup
                if args.cache_features:
                    print(f"Pre-caching CLIP features for {domain_name}...")
                    cache_name = f"{dataset_name}_{domain_name}_{args.backbone.replace('/', '_')}"
                    dataset = cache_features(dataset, clip_model, device=device,
                                            cache_dir='./cached_features', dataset_name=cache_name)
                
                # 切分数据给各客户端（随机打乱后不重叠切分，确保每客户端类别均匀）
                total_samples_domain = len(dataset)
                samples_per_client = total_samples_domain // args.num_clients
                client_datasets = OrderedDict()
                rng = np.random.RandomState(args.seed + domain_idx)  # 每个域用不同seed，但可复现
                shuffled = rng.permutation(total_samples_domain)
                for i in range(args.num_clients):
                    start_idx = i * samples_per_client
                    end_idx = start_idx + samples_per_client if i < args.num_clients - 1 else total_samples_domain
                    client_datasets[f'client_{i}'] = Subset(dataset, shuffled[start_idx:end_idx].tolist())

                print(f"Created {len(client_datasets)} clients for {domain_name}.")
                
                # 创建服务器并运行评估
                server = MccTtaServer(client_datasets, clip_weights, args, client_class=MccTtaClient)
                acc, total_correct, total_samples, stats = server.evaluate()
                
                domain_results[domain_name] = {
                    'accuracy': acc,
                    'correct': total_correct,
                    'total': total_samples
                }
                
                total_correct_all += total_correct
                total_samples_all += total_samples
                
                print(f"  {domain_name}: {acc * 100:.2f}% ({total_correct}/{total_samples})")
            
            # 打印汇总结果
            overall_acc = total_correct_all / total_samples_all
            print(f"\n{'='*60}")
            print(f"Results for {dataset_name}:")
            print(f"{'='*60}")
            for domain_name, result in domain_results.items():
                print(f"  {domain_name:20s}: {result['accuracy'] * 100:6.2f}%")
            print(f"  {'Total':20s}: {overall_acc * 100:6.2f}% ({total_correct_all}/{total_samples_all})")
            print(f"{'='*60}\n")
            
            if args.wandb:
                run_name = f"{dataset_name}_separate"
                run = wandb.init(project="MCC-TTA", config=cfg, group=group_name, name=run_name)
                for domain_name, result in domain_results.items():
                    wandb.log({f"{dataset_name}/{domain_name}": result['accuracy'] * 100})
                wandb.log({f"{dataset_name}/Total": overall_acc * 100})
                run.finish()

            if args.diagnostic_logger is not None:
                args.diagnostic_logger.write_outputs(
                    dataset_name=dataset_name,
                    backbone=args.backbone,
                    final_accuracy=overall_acc,
                    total_correct=total_correct_all,
                    total_samples=total_samples_all,
                )
                args.diagnostic_logger.print_summary_table(dataset_name, args.backbone)
            print_gaussian_noise_report(args)
        
        else:
            # 合并所有domain/corruption一起评估
            domain_loaders, classnames, template = build_test_data_loader(
                dataset_name, args.data_root, preprocess, separate_domains=True)
            clip_weights = clip_classifier(classnames, template, clip_model)
            setup_gaussian_noise_injector(args, clip_weights.shape[0])
            
            num_domains = len(domain_loaders)
            total_clients = num_domains * args.num_clients
            
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset_name} - Collaborative Mode")
            print(f"Strategy: Partition data first, then apply {num_domains} domains/corruptions")
            print(f"Total: {total_clients} clients ({num_domains} domains × {args.num_clients} clients)")
            print(f"{'='*60}\n")
            
            # Prepare datasets for each domain/corruption.
            # Cache/config reading is unchanged; only the client-index assignment changes below.
            prepared_domains = []
            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f"Setting up clients for {domain_name}...")
                dataset = domain_loader.dataset

                # Optional: cache features for speedup
                if args.cache_features:
                    cache_name = f"{dataset_name}_{domain_name}_{args.backbone.replace('/', '_')}"
                    dataset = cache_features(dataset, clip_model, device=device,
                                            cache_dir='./cached_features', dataset_name=cache_name)

                prepared_domains.append((domain_name, dataset))

            use_dirichlet_partition = getattr(args, 'client_partition', 'original') == 'dirichlet'
            # 为每个domain/corruption创建客户端。
            # full: 当前主实验协议，每个corruption完整切分；
            # nonoverlap: original-image index is assigned to only one corruption-client.
            all_client_datasets = OrderedDict()
            use_nonoverlap = is_cifar_c_dataset(dataset_name) and args.cifar_protocol == 'nonoverlap'

            if use_dirichlet_partition:
                if not is_cifar_c_dataset(dataset_name):
                    raise ValueError("--client_partition dirichlet is only supported for CIFAR-C datasets.")
                if not args.partition_file:
                    raise ValueError("--partition_file is required when --client_partition dirichlet.")

                from scripts.rebuttal_dirichlet_partition import apply_partition_file
                all_client_datasets, partition_meta = apply_partition_file(
                    args.partition_file,
                    prepared_domains,
                    dataset_name=dataset_name,
                    num_clients=args.num_clients,
                )
                print("Protocol: fixed Dirichlet label-skew partition")
                print(f"  partition file: {partition_meta['partition_file'] if 'partition_file' in partition_meta else args.partition_file}")
                print(f"  partition sha256: {partition_meta['partition_hash']}")
                print(f"  alpha: {partition_meta['dirichlet_alpha']}")
                print(f"  partition seed: {partition_meta['partition_seed']}")
                for domain_name, _ in prepared_domains:
                    domain_total = sum(len(all_client_datasets[f'{domain_name}_client_{i}']) for i in range(args.num_clients))
                    print(f"  {domain_name}: {args.num_clients} clients, {domain_total} samples total")
            elif use_nonoverlap:
                n_splits = num_domains * args.num_clients
                base_labels = extract_labels_for_partition(prepared_domains[0][1])
                folds = stratified_nonoverlap_folds(base_labels, n_splits=n_splits, seed=args.seed)
                print(f"Protocol: CIFAR-C non-overlap control")
                print(f"  Base indices: {len(base_labels)}; folds: {n_splits}; expected total evaluated samples: {sum(len(f) for f in folds)}")

                for domain_idx, (domain_name, dataset) in enumerate(prepared_domains):
                    for i in range(args.num_clients):
                        fold_id = domain_idx * args.num_clients + i
                        indices = folds[fold_id].tolist()
                        all_client_datasets[f'{domain_name}_client_{i}'] = Subset(dataset, indices)
                    domain_total = sum(len(folds[domain_idx * args.num_clients + i]) for i in range(args.num_clients))
                    print(f"  {domain_name}: {args.num_clients} clients, {domain_total} samples total (~{domain_total // args.num_clients}/client)")
            else:
                protocol_name = 'CIFAR-C full-corruption' if is_cifar_c_dataset(dataset_name) else 'domain full-stream'
                print(f"Protocol: {protocol_name}")
                for domain_idx, (domain_name, dataset) in enumerate(prepared_domains):
                    total_samples = len(dataset)
                    samples_per_client = total_samples // args.num_clients

                    # Full protocol: each domain/corruption stream is fully evaluated and split into clients.
                    rng = np.random.RandomState(args.seed + domain_idx)
                    indices_pool = rng.permutation(total_samples).tolist()

                    for i in range(args.num_clients):
                        start_idx = i * samples_per_client
                        end_idx = start_idx + samples_per_client if i < args.num_clients - 1 else total_samples
                        indices = indices_pool[start_idx:end_idx]
                        all_client_datasets[f'{domain_name}_client_{i}'] = Subset(dataset, indices)

                    print(f"  {domain_name}: {args.num_clients} clients, {total_samples} samples total (~{samples_per_client}/client)")

            print(f"\nCreated {len(all_client_datasets)} clients total.")
            print(f"Total evaluated samples after partition: {sum(len(ds) for ds in all_client_datasets.values())}")
            print(f"Participation rate: {args.part_rate} ({int(len(all_client_datasets) * args.part_rate)} clients per round)")

            if args.wandb:
                run_name = f"{dataset_name}_collaborative"
                run = wandb.init(project="MCC-TTA", config=cfg, group=group_name, name=run_name)

            # 创建服务器并运行评估
            server = MccTtaServer(all_client_datasets, clip_weights, args, client_class=MccTtaClient)
            acc, total_correct, total_samples, stats = server.evaluate()
            
            print(f"\n{'='*60}")
            print(f"Overall Results:")
            print(f"{'='*60}")
            print(f"Total Accuracy: {acc * 100:.2f}% ({total_correct}/{total_samples})")
            print(f"{'='*60}\n")
            
            # 按corruption统计结果
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
                    print(f"  {domain_name:20s}: {corruption_acc * 100:6.2f}% ({corruption_correct}/{corruption_total})")
            
            if args.wandb:
                wandb.log({f"{dataset_name}/Total": acc * 100})
                for domain_name, domain_acc in corruption_results.items():
                    wandb.log({f"{dataset_name}/{domain_name}": domain_acc * 100})
                run.finish()

            if use_dirichlet_partition:
                from scripts.rebuttal_dirichlet_partition import record_dirichlet_result
                runs_path, summary_path = record_dirichlet_result(
                    method="MCC-TTA",
                    dataset_name=dataset_name,
                    partition_file=args.partition_file,
                    accuracy=acc,
                    num_clients=args.num_clients,
                    num_samples=total_samples,
                )
                print(f"Saved Dirichlet run record: {runs_path.as_posix()}")
                print(f"Saved Dirichlet summary: {summary_path.as_posix()}")

            if args.diagnostic_logger is not None:
                args.diagnostic_logger.write_outputs(
                    dataset_name=dataset_name,
                    backbone=args.backbone,
                    final_accuracy=acc,
                    total_correct=total_correct,
                    total_samples=total_samples,
                )
                args.diagnostic_logger.print_summary_table(dataset_name, args.backbone)
            print_gaussian_noise_report(args)


if __name__ == "__main__":
    main()
