# -*- coding: utf-8 -*-
"""
TPT Runner - 多客户端并行 Test-Time Prompt Tuning
数据分配逻辑完全对齐 tda_runner.py，核心方法替换为 TPT（熵最小化更新 prompt）
"""

import random
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import clip

from utils import (
    get_config_file,
    build_test_data_loader,
    avg_entropy,
)


# ==================== 参数解析 ====================

def get_arguments():
    parser = argparse.ArgumentParser(description='TPT: Multi-Client Test-Time Prompt Tuning')
    parser.add_argument('--config',           dest='config',         required=True)
    parser.add_argument('--datasets',         dest='datasets',       type=str, required=True)
    parser.add_argument('--data-root',        dest='data_root',      type=str, default='./dataset/')
    parser.add_argument('--backbone',         dest='backbone',       type=str,
                        choices=['RN50', 'ViT-B/16'], required=True)
    parser.add_argument('--separate-domains', dest='separate_domains', action='store_true')
    parser.add_argument('--num-clients',      dest='num_clients',    type=int,   default=10)
    parser.add_argument('--part-rate',        dest='part_rate',      type=float, default=1.0)
    parser.add_argument('--sync-freq',        dest='sync_freq',      type=int,   default=100)
    parser.add_argument('--seed',                                    type=int,   default=1)
    parser.add_argument('--gpu',                                     type=int,   default=0)
    parser.add_argument('--cache-features',   dest='cache_features', action='store_true',
                        help='保留兼容性，TPT 不使用')
    parser.add_argument('--max-samples',      dest='max_samples',    type=int, default=None,
                        help='Profiling模式：最多评估的样本总数（跨所有客户端）。不设置则跑完整数据。')
    return parser.parse_args()


# ==================== GPU 批量 AugMix-like 多视图增强 ====================

class AugMixAugmenter:
    def __init__(self, augmix=True):
        self.augmix = augmix

    @torch.no_grad()
    def __call__(self, img_tensor, num_views, device, dtype):
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=torch.float32).view(1, 3, 1, 1)

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
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            x = (x - gray) * saturation.view(n, 1, 1, 1) + gray

        x = x.clamp(0.0, 1.0)
        x = (x - mean) / std
        return torch.cat([orig, x], dim=0).to(dtype=dtype)


# ==================== 可学习 Prompt ====================

class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init=None):
        super().__init__()
        device  = next(clip_model.parameters()).device
        dtype   = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_cls = len(classnames)

        if ctx_init and len(ctx_init.strip()) > 0:
            ctx_init_str = ctx_init.replace('_', ' ').strip()
            n_ctx = len(ctx_init_str.split())
            with torch.no_grad():
                tok = clip.tokenize(ctx_init_str).to(device)
                emb = clip_model.token_embedding(tok).type(dtype)
            ctx_vectors = emb[0, 1: 1 + n_ctx, :].detach().clone().float()
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.n_ctx = n_ctx
        self.dtype = dtype
        self.ctx   = nn.Parameter(ctx_vectors)
        self.ctx_init_state = ctx_vectors.detach().clone()

        classnames_clean = [c.replace('_', ' ') for c in classnames]
        prompts   = ['a photo of a {}.'.format(c) for c in classnames_clean]
        tokenized = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(dtype)

        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])
        self.tokenized_prompts = tokenized

    def forward(self):
        ctx = self.ctx.to(dtype=self.dtype).unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

    def reset(self):
        with torch.no_grad():
            self.ctx.copy_(self.ctx_init_state)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer          = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final             = clip_model.ln_final
        self.text_projection      = clip_model.text_projection
        self.dtype                = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eos_idx = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eos_idx]
        x = x @ self.text_projection
        return x


class TPTModel(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init=None):
        super().__init__()
        self.image_encoder  = clip_model.visual
        self.prompt_learner = PromptLearner(clip_model, classnames, n_ctx, ctx_init)
        self.text_encoder   = TextEncoder(clip_model)
        self.logit_scale    = clip_model.logit_scale
        self.dtype          = clip_model.dtype
        for name, param in self.named_parameters():
            if 'prompt_learner' not in name:
                param.requires_grad_(False)

    @torch.no_grad()
    def encode_images(self, image):
        """只算 image features，无梯度，供 tta 循环复用"""
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_text_features(self):
        """只算 text features，有梯度（prompt 可学习）"""
        prompts       = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, image):
        image_features = self.encode_images(image)
        text_features  = self.get_text_features()
        logits = self.logit_scale.exp() * image_features @ text_features.T
        return logits

    def forward_with_image_features(self, image_features):
        """用预计算的 image features 直接算 logits，避免重复跑 image encoder"""
        text_features = self.get_text_features()
        logits = self.logit_scale.exp() * image_features @ text_features.T
        return logits

    def reset(self):
        self.prompt_learner.reset()


# ==================== TPT 核心 ====================

def select_confident_samples(logits, top):
    """原论文 select_confident_samples：按熵升序保留前 top 比例"""
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    k = max(1, int(logits.size(0) * top))
    idx = torch.argsort(batch_entropy, descending=False)[:k]
    return logits[idx], idx


class TPTClient:
    """
    单个客户端，接口完全对齐 tda_runner.BaseClient：
    - curr_idx 推进
    - is_done() 用 curr_idx >= num_samples
    - evaluate_one() 返回 (correct, 1)
    """
    def __init__(self, dataset, tpt_model, optimizer, optim_state, tpt_cfg, device):
        self.dataset     = dataset
        self.num_samples = len(dataset)
        self.curr_idx    = 0
        self.num_correct = 0
        self.tpt_model   = tpt_model
        self.optimizer   = optimizer
        self.optim_state = optim_state
        self.tpt_cfg     = tpt_cfg
        self.device      = device

        # 缓存 ctx 初始状态，用于快速 reset（避免 load_state_dict 开销）
        self._ctx_init = tpt_model.prompt_learner.ctx_init_state.clone()

        self.n_views = tpt_cfg.get('n_views', 64)
        self.augmix = tpt_cfg.get('augmix', True)
        self.augmenter = AugMixAugmenter(augmix=self.augmix)

    def _fast_reset(self):
        """直接 copy_ 覆盖 ctx，比 load_state_dict 快 10x"""
        with torch.no_grad():
            self.tpt_model.prompt_learner.ctx.copy_(self._ctx_init)
        # 清空 optimizer 动量（直接清零，不 reload state_dict）
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                    state['exp_avg_sq'].zero_()
                    state['step'] = torch.zeros_like(state['step']) if torch.is_tensor(state.get('step')) else 0

    def evaluate_one(self):
        """对齐 tda_runner.BaseClient.evaluate_one：用 curr_idx 取样本"""
        if self.curr_idx >= self.num_samples:
            return 0, 0

        data, label = self.dataset[self.curr_idx]
        self.curr_idx += 1
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)
        if not torch.is_tensor(data):
            raise TypeError('TPT GPU AugMix expects preprocessed image tensors. Build data loader with CLIP preprocess.')
        images_aug = self.augmenter(
            data, num_views=self.n_views,
            device=self.device, dtype=self.tpt_model.dtype)

        tta_steps   = self.tpt_cfg.get('tta_steps',   1)
        selection_p = self.tpt_cfg.get('selection_p', 0.1)

        # 快速 reset：直接 copy_ 而非 load_state_dict
        if len(self.optimizer.state) == 0:
            self.tpt_model.reset()
        else:
            self._fast_reset()

        # 关键加速：image features 只算一次（image encoder 无梯度，不随 prompt 变化）
        # 注意：AugMixAugmenter 返回的第一个视图就是原图，因此无需再单独 encode 一次原图
        image_features_aug  = self.tpt_model.encode_images(images_aug)   # [N, D]
        image_features_orig = image_features_aug[:1]                     # [1, D]

        # tta_steps 步熵最小化（只重算 text features）
        selected_idx = None
        for _ in range(tta_steps):
            output = self.tpt_model.forward_with_image_features(image_features_aug)
            if selected_idx is None:
                output, selected_idx = select_confident_samples(output, selection_p)
            else:
                output = output[selected_idx]
            loss = avg_entropy(output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if not torch.isfinite(self.tpt_model.prompt_learner.ctx).all():
                self.tpt_model.reset()
                raise FloatingPointError('TPT prompt became non-finite during adaptation. Try larger AdamW eps or lower lr.')

        with torch.no_grad():
            final_logits = self.tpt_model.forward_with_image_features(image_features_orig)
        pred = int(final_logits.argmax(dim=-1).item())

        correct = int(pred == label_int)
        self.num_correct += correct
        return correct, 1

    def is_done(self):
        """对齐 tda_runner.BaseClient.is_done"""
        return self.curr_idx >= self.num_samples


class TPTServer:
    """
    调度逻辑完全对齐 tda_runner.BaseCTTAServer.evaluate()
    """
    def __init__(self, datasets, tpt_model, tpt_cfg, device, part_rate=1.0, sync_freq=100, max_samples=None):
        lr = tpt_cfg.get('lr', 5e-3)
        self.optimizer   = torch.optim.AdamW(tpt_model.prompt_learner.parameters(), lr=lr, eps=1e-4)
        self.optim_state = deepcopy(self.optimizer.state_dict())

        self.clients = OrderedDict()
        for cid, dataset in datasets.items():
            self.clients[cid] = TPTClient(
                dataset, tpt_model, self.optimizer, self.optim_state, tpt_cfg, device)

        self.client_ids  = list(self.clients.keys())
        self.num_clients = len(self.clients)
        self.cohort_size = max(1, int(self.num_clients * part_rate))
        self.num_rounds  = sum(c.num_samples for c in self.clients.values())
        self.sync_freq   = sync_freq
        self.max_samples = max_samples

    def synchronize(self):
        pass  # 所有客户端共享同一个 tpt_model，天然同步

    def evaluate(self):
        """完全对齐 tda_runner.BaseCTTAServer.evaluate()"""
        total_correct, total_num_samples = 0, 0
        done_count = 0  # 已完成客户端计数，避免每轮遍历所有客户端

        for rnd in tqdm(range(1, self.num_rounds + 1), desc='TPT Federated TTA'):

            # Profiling 模式：达到样本上限后提前停止
            if self.max_samples is not None and total_num_samples >= self.max_samples:
                break

            # 每轮随机选 cohort_size 个客户端（对齐 tda_runner randperm 逻辑）
            selected_idx = sorted(
                torch.randperm(self.num_clients)[:self.cohort_size].tolist())

            rnd_correct, rnd_num_samples = 0, 0
            for idx in selected_idx:
                if self.max_samples is not None and total_num_samples + rnd_num_samples >= self.max_samples:
                    break
                client = self.clients[self.client_ids[idx]]
                was_done = client.is_done()
                correct, num = client.evaluate_one()
                # 检测本轮是否刚完成
                if not was_done and client.is_done():
                    done_count += 1
                rnd_correct     += correct
                rnd_num_samples += num
            total_correct     += rnd_correct
            total_num_samples += rnd_num_samples


            # 所有客户端完成则提前退出
            if done_count >= self.num_clients:
                break

            if rnd % self.sync_freq == 0:
                self.synchronize()

        stats = {cid: client.num_correct / client.curr_idx if client.curr_idx > 0 else 0.0
                 for cid, client in self.clients.items()}
        overall_acc = total_correct / total_num_samples if total_num_samples > 0 else 0.0
        return overall_acc, total_correct, total_num_samples, stats


# ==================== 主函数 ====================

def main():
    args = get_arguments()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
    print(f'Using device: {device}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    clip_model_raw, preprocess_clip = clip.load(args.backbone, device=device)
    clip_model_raw.eval()
    print(f'Loaded CLIP backbone: {args.backbone}')

    _name_map = {
        'VLCS':          'vlcs',
        'TerraIncognita':'terra',
        'OfficeHome':    'officehome',
        'CIFAR10CFull':  'cifar10c',
        'CIFAR100CFull': 'cifar100c',
    }
    _cifar_datasets = {'CIFAR10CFull', 'CIFAR100CFull'}

    for dataset_name in args.datasets.split('/'):
        print(f'\n{"="*60}')
        print(f'Processing dataset: {dataset_name}')
        print(f'{"="*60}')

        _suffix      = _name_map.get(dataset_name, dataset_name.lower())
        _config_file = os.path.join(args.config, f'tpt_{_suffix}.yaml')
        if not os.path.exists(_config_file):
            _config_file = os.path.join(os.path.dirname(__file__), 'configs', f'tpt_{_suffix}.yaml')
        if os.path.exists(_config_file):
            with open(_config_file, 'r', encoding='utf-8') as _f:
                cfg = yaml.load(_f, Loader=yaml.SafeLoader)
        else:
            cfg = get_config_file(args.config, dataset_name)
        tpt_cfg = cfg.get('tpt', cfg)
        print(f'TPT config: {tpt_cfg}')

        if args.separate_domains:
            # 分域模式：每个 domain 内分割给 num_clients 个客户端
            domain_loaders, classnames, template = build_test_data_loader(
                dataset_name, args.data_root, preprocess_clip, separate_domains=True)

            domain_results    = {}
            total_correct_all = 0
            total_samples_all = 0

            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                print(f'\n--- Evaluating domain: {domain_name} ---')
                dataset = domain_loader.dataset
                total_samples_domain = len(dataset)
                samples_per_client   = total_samples_domain // args.num_clients
                rng = np.random.RandomState(args.seed + domain_idx)
                shuffled = rng.permutation(total_samples_domain).tolist()

                client_datasets = OrderedDict()
                for i in range(args.num_clients):
                    start = i * samples_per_client
                    end   = start + samples_per_client if i < args.num_clients - 1 else total_samples_domain
                    client_datasets[f'client_{i}'] = Subset(dataset, shuffled[start:end])

                print(f'  Created {args.num_clients} clients, ~{samples_per_client} samples each')

                tpt_model = TPTModel(clip_model_raw, classnames,
                                     n_ctx=tpt_cfg.get('n_ctx', 4),
                                     ctx_init=tpt_cfg.get('ctx_init', None))
                tpt_model = tpt_model.to(device)

                server = TPTServer(client_datasets, tpt_model, tpt_cfg, device,
                                   part_rate=args.part_rate, sync_freq=args.sync_freq,
                                   max_samples=args.max_samples)
                acc, total_correct, total_samples, stats = server.evaluate()

                domain_results[domain_name] = acc
                total_correct_all += total_correct
                total_samples_all += sum(client.curr_idx for client in server.clients.values())
                print(f'  {domain_name}: {acc * 100:.2f}% ({total_correct}/{total_samples})')

            overall_acc = total_correct_all / total_samples_all if total_samples_all > 0 else 0.0
            print(f'\n{"="*60}')
            print(f'Overall Results ({dataset_name}):')
            for dname, dacc in domain_results.items():
                print(f'  {dname:20s}: {dacc * 100:.2f}%')
            print(f'  {"Average":20s}: {overall_acc * 100:.2f}%')
            print(f'{"="*60}\n')

        else:
            # 合并模式：每个 domain 分割给 num_clients 个客户端
            domain_loaders, classnames, template = build_test_data_loader(
                dataset_name, args.data_root, preprocess_clip, separate_domains=True)

            num_domains   = len(domain_loaders)
            total_clients = num_domains * args.num_clients
            print(f'\nCollaborative Mode: {num_domains} domains x {args.num_clients} clients = {total_clients} total')
            print(f'Participation rate: {args.part_rate} ({int(total_clients * args.part_rate)} clients per round)')

            all_client_datasets = OrderedDict()
            for domain_idx, (domain_name, domain_loader) in enumerate(domain_loaders.items()):
                dataset = domain_loader.dataset
                total_samples      = len(dataset)
                samples_per_client = total_samples // args.num_clients
                if dataset_name in _cifar_datasets:
                    rng = np.random.RandomState(args.seed + domain_idx)
                    indices_pool = rng.permutation(total_samples).tolist()
                else:
                    rng = np.random.RandomState(args.seed + domain_idx)
                    indices_pool = rng.permutation(total_samples).tolist()
                for i in range(args.num_clients):
                    start = i * samples_per_client
                    end   = start + samples_per_client if i < args.num_clients - 1 else total_samples
                    all_client_datasets[f'{domain_name}_client_{i}'] = Subset(
                        dataset, indices_pool[start:end])
                print(f'  {domain_name}: {args.num_clients} clients, ~{samples_per_client} samples each')


            tpt_model = TPTModel(clip_model_raw, classnames,
                                 n_ctx=tpt_cfg.get('n_ctx', 4),
                                 ctx_init=tpt_cfg.get('ctx_init', None))
            tpt_model = tpt_model.to(device)

            server = TPTServer(all_client_datasets, tpt_model, tpt_cfg, device,
                               part_rate=args.part_rate, sync_freq=args.sync_freq,
                               max_samples=args.max_samples)
            acc, total_correct, total_samples, stats = server.evaluate()

            print(f'\n{"="*60}')
            print(f'Overall Results ({dataset_name}):')
            print(f'Total Accuracy: {acc * 100:.2f}% ({total_correct}/{total_samples})')
            for domain_name in domain_loaders.keys():
                d_correct = sum(server.clients[f'{domain_name}_client_{i}'].num_correct
                                for i in range(args.num_clients)
                                if f'{domain_name}_client_{i}' in server.clients)
                d_total   = sum(server.clients[f'{domain_name}_client_{i}'].curr_idx
                                for i in range(args.num_clients)
                                if f'{domain_name}_client_{i}' in server.clients)
                if d_total > 0:
                    print(f'  {domain_name:20s}: {d_correct/d_total*100:.2f}% ({d_correct}/{d_total})')
            print(f'{"="*60}\n')


if __name__ == '__main__':
    main()

