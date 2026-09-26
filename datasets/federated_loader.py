"""
Federated Dataset Loader
用于将数据集分配给多个客户端
"""

import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from tqdm import tqdm


class FederatedDataset:
    """联邦数据集：将数据集分配给多个客户端"""
    
    def __init__(self, dataset, num_clients, partition='iid', seed=42):
        """
        初始化联邦数据集
        :param dataset: 原始数据集
        :param num_clients: 客户端数量
        :param partition: 分区方式 ('iid', 'domain', 'corruption')
        :param seed: 随机种子
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition = partition
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.client_datasets = self._partition_data()
    
    def _partition_data(self):
        """根据分区方式划分数据"""
        if self.partition == 'iid':
            return self._partition_iid()
        elif self.partition == 'domain':
            return self._partition_by_domain()
        elif self.partition == 'corruption':
            return self._partition_by_corruption()
        else:
            raise ValueError(f"Unknown partition method: {self.partition}")
    
    def _partition_iid(self):
        """IID分区：随机均匀分配"""
        num_samples = len(self.dataset)
        indices = np.random.permutation(num_samples)
        
        client_datasets = OrderedDict()
        samples_per_client = num_samples // self.num_clients
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else num_samples
            client_indices = indices[start_idx:end_idx]
            client_datasets[f'client_{i}'] = Subset(self.dataset, client_indices)
        
        return client_datasets
    
    def _partition_by_domain(self):
        """按域分区：假设数据集有domain属性"""
        # 这个方法需要根据具体数据集实现
        # 例如VLCS有4个域，每个域分配给若干客户端
        raise NotImplementedError("Domain partition needs dataset-specific implementation")
    
    def _partition_by_corruption(self):
        """
        按污染类型分区：CIFAR-C数据集
        假设dataset是MultipleCorruptionNumpyImageDataset，包含多个corruption子数据集
        每个corruption分配给 num_clients_per_corruption 个客户端
        """
        # 检查数据集是否有多个子数据集（corruption）
        if not hasattr(self.dataset, 'datasets'):
            raise ValueError("Dataset must have 'datasets' attribute for corruption partition")
        
        corruption_datasets = self.dataset.datasets
        num_corruptions = len(corruption_datasets)
        
        # 计算每个corruption分配多少个客户端
        clients_per_corruption = self.num_clients // num_corruptions
        
        if clients_per_corruption == 0:
            raise ValueError(f"num_clients ({self.num_clients}) must be >= num_corruptions ({num_corruptions})")
        
        print(f"Corruption partition: {num_corruptions} corruptions, {clients_per_corruption} clients per corruption")
        
        client_datasets = OrderedDict()
        client_id = 0
        
        # 为每个corruption创建多个客户端
        for corruption_idx, corruption_dataset in enumerate(corruption_datasets):
            num_samples = len(corruption_dataset)
            indices = np.random.permutation(num_samples)
            
            samples_per_client = num_samples // clients_per_corruption
            
            for i in range(clients_per_corruption):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client if i < clients_per_corruption - 1 else num_samples
                client_indices = indices[start_idx:end_idx]
                
                # 创建子数据集
                client_dataset = Subset(corruption_dataset, client_indices)
                client_datasets[f'corruption_{corruption_idx}_client_{i}'] = client_dataset
                client_id += 1
        
        return client_datasets
    
    def get_client_datasets(self):
        """返回客户端数据集字典"""
        return self.client_datasets


class CachedDataset(Dataset):
    """缓存数据集：预先提取CLIP特征"""
    
    def __init__(self, features, labels):
        """
        :param features: 预提取的特征 (N * feat_dim)
        :param labels: 标签 (N,)
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx].unsqueeze(0), self.labels[idx]


def cache_features(dataset, clip_model, device='cuda', batch_size=64, cache_dir='./cached_features', dataset_name=None):
    """
    预提取CLIP特征（支持保存和加载）
    :param dataset: 原始数据集
    :param clip_model: CLIP模型
    :param device: 设备
    :param batch_size: 批大小
    :param cache_dir: 缓存目录
    :param dataset_name: 数据集名称（用于生成缓存文件名）
    :return: CachedDataset
    """
    import os
    from torch.utils.data import DataLoader
    
    # 如果提供了dataset_name，尝试加载缓存
    if dataset_name is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{dataset_name}.pt')
        
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            cached_data = torch.load(cache_file)
            return CachedDataset(cached_data['features'], cached_data['labels'])
    
    # 缓存不存在，重新提取特征
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    all_labels = []
    
    clip_model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Caching features"):
            images = images.to(device)
            features = clip_model.encode_image(images)
            features = F.normalize(features, dim=-1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 保存缓存
    if dataset_name is not None:
        print(f"Saving cached features to {cache_file}")
        torch.save({'features': all_features, 'labels': all_labels}, cache_file)
    
    return CachedDataset(all_features, all_labels)
