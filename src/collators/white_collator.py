import torch 
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List
import numpy as np
import torch.nn.functional as F
import sys 
sys.path.append("/root/TracGPT-R3D")
from src.model.bbox3d.helper import corners_to_center,center_to_corners,get_center
from collections import defaultdict


status_map={
    "Status (Non-Dementia)":0,
    "Status (Mild-Dementia)":1,
    "Status (Moderate-Dementia)":2
}

class WhiteCollator:
    def __call__(self, batch):
        images = []
        bbox_metrics = defaultdict(list)
        status_targets = []
        for sample in batch:
            images.append(sample['image'])
            status_targets.append(status_map[sample['A4']])
            for metric, val in sample["A3"].items():
                bbox_metrics[metric].append(val)
        
        processed = {
            "images": torch.stack(images),  # [B, C, H, W]
            "status_criteria": torch.tensor(status_targets).long()  # [B]
        }
        
        for metric, values in bbox_metrics.items():
            processed[f"bbox_{metric}"] = torch.tensor(values).float()  # [B, ...]
        
        return processed