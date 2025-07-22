import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from types import SimpleNamespace
from collections import OrderedDict, defaultdict
import sys
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
import torch
from model.bbox3d.bbox_head import BBox3DHead,AnchorBBox3DHead,AnchorBBox3DHeadV2
from model.bbox3d.helper import hungarian_iou_matching,convert_model_to_gt_format
from model.bbox3d.helper import box3d_iou_single,iou_loss,diou_3d

class AnchorBBox3DHeadV2(nn.Module):
    def __init__(self,config,num_anchors=3,patch_grid=[4,4,4],embed_dim=768,in_channels=64):
        super().__init__()
        # in_channels= config.in_channels if isinstance(config, SimpleNamespace) else in_channels
        self.max_pred_per_patch=3
        self.num_anchors = num_anchors
        self.downsample = nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.patch_grid=patch_grid
        self.head_dim=self.max_pred_per_patch*7
        self.reducer=nn.Sequential(
            nn.Linear(768, 256), 
            nn.ReLU(),           
            nn.Linear(256, 1)    
        )
        self.bbox_reducer=nn.Sequential(
            nn.Linear(768, 256), 
            nn.ReLU(),           
            nn.Linear(256,  self.head_dim)    
        )
    


    def forward(self, x):
        # input: (B, 64,768)
        head_dim=self.max_pred_per_patch*7
        B,all_patches,embed = x.shape[0],x.shape[1],x.shape[2]  # Batch size
        
        center_pred=torch.sigmoid(self.reducer(x)).squeeze(-1)   # [B, 64,1]
        threshold = 0.5
        # print()
        center_pred = torch.where(
            center_pred > threshold,
            torch.ones_like(center_pred),  # Values > threshold → 1
            torch.zeros_like(center_pred)   # Values ≤ threshold → 0
        )
        # print("bin shape",center_pred.shape)
        print("unique value",torch.unique(center_pred))
        center_pred=center_pred.reshape(B,self.patch_grid[0],self.patch_grid[1],self.patch_grid[2])
        center_pred_bool=center_pred.bool()
        # print("x shape",x.shape)
        print("unique value",torch.unique(center_pred))

        x_split=x.reshape(B,self.patch_grid[0],self.patch_grid[1],self.patch_grid[2],embed)
        x_head=self.bbox_reducer(x_split)
        x_head=x_head.view(B,x_head.shape[1],x_head.shape[2],x_head.shape[3],3,-1)
        delta_xyz = torch.sigmoid(x_head[:, :,  :, :, :,:3]) * 2 - 0.5  # [2, 3, 3, 16, 32, 32]
        log_dwh = x_head[:, :,  :, :, :,3:6]  # [2, 3, 3, 16, 32, 32]
        conf = torch.sigmoid(x_head[:, :,  :, :, :,6])  # [2, 3, 16, 32, 32]
        # print("x_head",x_head.shape)
        
        return center_pred_bool,delta_xyz,log_dwh,conf
    
class AnchorBBox3DLossV2(nn.Module):
    def __init__(self,  pos_weight=1.0, neg_weight=0.5, lambda_reg=1.0):
        super().__init__()
        print("V2 anchor bbox loss")
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.lambda_reg = lambda_reg
        
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none')  # Works with logits

    def forward(self, delta_zxy, log_dwh, conf_pred, gt_boxes, masks):
        """
        Args:
            delta_zxy: (B, D, H, W, N_boxes, 3) - Position offsets
            log_dwh: (B, D, H, W, N_boxes, 3) - Log-dimensions
            conf_pred: (B, D, H, W, N_boxes) - Confidence logits
            gt_boxes: (B, max_objs, 6) - Normalized boxes (cx,cy,cz,w,h,d)
            masks: (B,max_objs)
        """
        B, D, H, W, num_boxes, _ = delta_zxy.shape
        device = delta_zxy.device

        # --- 1. Create grid centers ---
        z_grid = torch.linspace(0.5/D, 1-0.5/D, D, device=device).view(1, D, 1, 1, 1)
        y_grid = torch.linspace(0.5/H, 1-0.5/H, H, device=device).view(1, 1, H, 1, 1)
        x_grid = torch.linspace(0.5/W, 1-0.5/W, W, device=device).view(1, 1, 1, W, 1)

        pos_target = torch.zeros_like(delta_zxy)
        size_target = torch.zeros_like(log_dwh)
        conf_target = torch.zeros_like(conf_pred)

        for b in range(B):
            print("gt_boxes", gt_boxes.shape, masks.shape)
            gt = gt_boxes[b][masks[b]]  # (N_gt, 6)
            
            if len(gt) == 0:
                continue  

            # Convert GT to grid indices
            grid_z = (gt[:, 0] * D).clamp(0, D-1).long()
            grid_y = (gt[:, 1] * H).clamp(0, H-1).long()
            grid_x = (gt[:, 2] * W).clamp(0, W-1).long()

            # Assign to all boxes at each spatial location
            for gt_idx in range(len(gt)):
                z, y, x = grid_z[gt_idx], grid_y[gt_idx], grid_x[gt_idx]
                
                # Position targets (offsets from grid centers)
                pos_target[b, z, y, x, :, 0] = gt[gt_idx, 0] - z_grid[0, z, 0, 0, 0]  # dz
                pos_target[b, z, y, x, :, 1] = gt[gt_idx, 1] - y_grid[0, 0, y, 0, 0]  # dy
                pos_target[b, z, y, x, :, 2] = gt[gt_idx, 2] - x_grid[0, 0, 0, x, 0]  # dx
                
                # Size targets (log of dimensions)
                size_target[b, z, y, x, :, 0] = torch.log(gt[gt_idx, 3] + 1e-6)  # log_w
                size_target[b, z, y, x, :, 1] = torch.log(gt[gt_idx, 4] + 1e-6)  # log_h
                size_target[b, z, y, x, :, 2] = torch.log(gt[gt_idx, 5] + 1e-6)  # log_d
                
                conf_target[b, z, y, x, :] = 1.0

        # Regression mask (positive anchors)
        reg_mask = conf_target > 0.5
        
        # Position loss
        pos_loss = self.reg_loss(delta_zxy[reg_mask], pos_target[reg_mask]).sum()
        
        # Size loss
        size_loss = self.reg_loss(log_dwh[reg_mask], size_target[reg_mask]).sum()
        
        # Confidence loss (weighted)
        conf_loss = (self.conf_loss(conf_pred, conf_target) * 
                   torch.where(conf_target > 0.5, self.pos_weight, self.neg_weight)).sum()
        
        # Normalize by number of positive anchors
        num_pos = max(1.0, reg_mask.float().sum())
        
        total_loss = (
            self.lambda_reg * (pos_loss + size_loss) / num_pos +
            conf_loss / num_pos
        )

        return {
            'total_loss': total_loss,
            'pos_loss': pos_loss / num_pos,
            'size_loss': size_loss / num_pos,
            'conf_loss': conf_loss / num_pos
        }
        