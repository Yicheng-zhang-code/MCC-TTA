"""
Dataset utility functions used by MCC-TTA.
Only the required helpers are included.
"""

import os
import os.path as osp
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image


def listdir_nohidden(path, sort=False):
    """列出目录中的非隐藏文件
    
    Args:
        path (str): 目录路径
        sort (bool): 是否排序
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def read_json(fpath):
    """从路径读取 JSON 文件"""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """写入 JSON 文件"""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """使用 PIL.Image 读取图像
    
    Args:
        path (str): 图像路径
        
    Returns:
        PIL Image (RGB)
    """
    if not osp.exists(path):
        raise IOError(f'No file exists at {path}')
    
    try:
        img = Image.open(path).convert('RGB')
        return img
    except IOError:
        raise IOError(f'Cannot read image from {path}')
