# Latte 数据集
from .corruption import CIFAR10CFull, CIFAR100CFull, get_corruption_dataset_class
from .domainbed import VLCS, TerraIncognita, get_domainbed_dataset_class


dataset_list = {
    # Corruption datasets (CIFAR-10-C, CIFAR-100-C)
    "CIFAR10CFull": CIFAR10CFull,
    "CIFAR100CFull": CIFAR100CFull,
    
    # DomainBed datasets (VLCS, TerraIncognita)
    "VLCS": VLCS,
    "TerraIncognita": TerraIncognita,
                }


def build_dataset(dataset, root_path):
    """根据数据集名称构建数据集实例"""
    if dataset not in dataset_list:
        raise ValueError(f"Dataset '{dataset}' not found. Available datasets: {list(dataset_list.keys())}")
    return dataset_list[dataset](root_path)