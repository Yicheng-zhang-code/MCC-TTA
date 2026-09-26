import os
import yaml
import torch
import math
import numpy as np
import clip
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        device = next(clip_model.parameters()).device  # 跟随 clip_model 所在设备

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=0).to(device)  # [num_class, feat_dim]
    return clip_weights


def get_clip_logits(image_feature, clip_weights, normalize=False, p=0.1, return_features=False):
    """
    Get CLIP logits and predictions (原论文版本).
    
    Args:
        image_feature: 已提取的 CLIP 特征 [batch_size, feat_dim]，已归一化
        clip_weights: 文本权重 [num_class, feat_dim]
        normalize: 是否再次归一化
        p: 批处理时的采样比例
        return_features: 是否返回特征
    
    Returns:
        如果 return_features=True: (clip_logits, pred, proba, entropy, image_feature)
        否则: (clip_logits, pred, proba, entropy)
    """
    with torch.no_grad():
        if normalize:
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            clip_weights = clip_weights / clip_weights.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_feature @ clip_weights.T
        
        batch_size = image_feature.shape[0]

        if batch_size > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_size * p)]
            output = clip_logits[selected_idx]
            image_features_selected = image_feature[selected_idx].mean(0).unsqueeze(0)
            image_features_selected = image_features_selected / image_features_selected.norm(dim=-1, keepdim=True)
            clip_logits = output.mean(0).unsqueeze(0)

            entropy = avg_entropy(output)
            proba = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
            
            if return_features:
                entropy_val = entropy.item() if torch.is_tensor(entropy) else entropy
                return clip_logits, pred, proba, entropy_val, image_features_selected
            else:
                return clip_logits, pred, proba, entropy
        else:
            entropy = softmax_entropy(clip_logits)
            proba = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])
            
            if return_features:
                entropy_val = entropy.item() if torch.is_tensor(entropy) else entropy
                return clip_logits, pred, proba, entropy_val, image_feature
            else:
                return clip_logits, pred, proba, entropy


def get_entropy_batch(image_features, clip_weights, normalize=False):
    """Compute entropy for a batch of image features."""
    with torch.no_grad():
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            clip_weights = clip_weights / clip_weights.norm(dim=-1, keepdim=True)
        
        # clip_weights: [num_class, feat_dim]
        # image_features: [batch, feat_dim]
        # result: [batch, num_class]
        clip_logits = 100.0 * image_features @ clip_weights.T
        entropy = softmax_entropy(clip_logits)
        return entropy


def get_preprocess():
    """获取标准的 CLIP 预处理"""
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess


def get_config_file(config_path, dataset_name):
    """
    加载数据集配置文件
    
    支持的数据集：
    - VLCS, TerraIncognita (DomainBed)
    - CIFAR10CFull, CIFAR100CFull (Corruption)
    """
    if config_path.endswith((".yaml", ".yml")):
        config_file = config_path
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as file:
            return yaml.load(file, Loader=yaml.SafeLoader)

    # Dataset-to-config mapping for MCC-TTA.
    config_mapping = {
        'VLCS': 'mcc_tta_vlcs.yaml',
        'TerraIncognita': 'mcc_tta_terra.yaml',
        'OfficeHome': 'mcc_tta_officehome.yaml',
        'CIFAR10CFull': 'mcc_tta_cifar10c.yaml',
        'CIFAR100CFull': 'mcc_tta_cifar100c.yaml',
    }
    
    # 获取配置文件名
    if dataset_name in config_mapping:
        config_name = config_mapping[dataset_name]
    else:
        config_name = f"mcc_tta_{dataset_name.lower()}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess, separate_domains=False):
    """
    构建测试数据加载器
    
    Supported MCC-TTA datasets:
    - VLCS, TerraIncognita (DomainBed 数据集)
    - CIFAR10CFull, CIFAR100CFull (Corruption 数据集)
    
    Args:
        dataset_name: 数据集名称
        root_path: 数据集根目录
        preprocess: 预处理函数
        separate_domains: 是否分别返回每个子数据集（用于单独评估）
    
    Returns:
        如果 separate_domains=False: (test_loader, classnames, template)
        如果 separate_domains=True: (domain_loaders_dict, classnames, template)
    """
    
    # MCC-TTA: CIFAR-10-C / CIFAR-100-C corruption datasets.
    if dataset_name in ['CIFAR10CFull', 'CIFAR100CFull']:
        from datasets.corruption import CIFAR10CFull, CIFAR100CFull
        
        # 创建数据集实例
        if dataset_name == 'CIFAR10CFull':
            dataset = CIFAR10CFull(root_path, extra=True, severity=5, transform=preprocess)
        else:
            dataset = CIFAR100CFull(root_path, extra=True, severity=5, transform=preprocess)
        
        classnames = dataset.classes
        template = [
            "itap of a {}",
            "a bad photo of the {}",
            "a origami {}",
            "a photo of the large {}",
            "a {} in a video game",
            "art of the {}",
            "a photo of the small {}"
        ]
        
        if separate_domains:
            # 返回每个损坏类型的独立 DataLoader
            domain_loaders = {}
            for i in range(len(dataset)):
                corruption_name = dataset.environments[i]
                domain_dataset = dataset[i]
                domain_loader = DataLoader(domain_dataset, batch_size=1, shuffle=False, num_workers=4)
                domain_loaders[corruption_name] = domain_loader
            return domain_loaders, classnames, template
        else:
            # 合并所有损坏类型
            all_datasets = [dataset[i] for i in range(len(dataset))]
            combined_dataset = ConcatDataset(all_datasets)
            test_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True, num_workers=4)
            return test_loader, classnames, template
    
    # MCC-TTA: VLCS / TerraIncognita / OfficeHome DomainBed datasets.
    elif dataset_name in ['VLCS', 'TerraIncognita', 'OfficeHome']:
        from datasets.domainbed import VLCS, TerraIncognita
        
        # 创建数据集实例
        if dataset_name == 'VLCS':
            dataset = VLCS(root_path, transform=preprocess)
        elif dataset_name == 'TerraIncognita':
            dataset = TerraIncognita(root_path, transform=preprocess)
        else:
            from datasets.domainbed import OfficeHome
            dataset = OfficeHome(root_path, transform=preprocess)
        
        classnames = dataset.classes
        template = [
            "itap of a {}",
            "a bad photo of the {}",
            "a origami {}",
            "a photo of the large {}",
            "a {} in a video game",
            "art of the {}",
            "a photo of the small {}"
        ]
        
        if separate_domains:
            # 返回每个领域的独立 DataLoader
            domain_loaders = {}
            for i in range(len(dataset)):
                domain_name = dataset.environments[i]
                domain_dataset = dataset[i]
                domain_loader = DataLoader(domain_dataset, batch_size=1, shuffle=False, num_workers=4)
                domain_loaders[domain_name] = domain_loader
            return domain_loaders, classnames, template
        else:
            # 合并所有领域
            all_datasets = [dataset[i] for i in range(len(dataset))]
            combined_dataset = ConcatDataset(all_datasets)
            test_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True, num_workers=4)
            return test_loader, classnames, template
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. "
                        f"Supported datasets: VLCS, TerraIncognita, OfficeHome, CIFAR10CFull, CIFAR100CFull")
