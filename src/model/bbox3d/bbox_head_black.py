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

class AnchorBBox3DHead(nn.Module):
    def __init__(self,config,num_anchors=3):
        super().__init__()
        in_channels= config.in_channels if isinstance(config, SimpleNamespace) else in_channels
        self.num_anchors = num_anchors
        self.downsample = nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv3d(
            in_channels, num_anchors * 9, kernel_size=1
        )  # 9 = (Δz,Δx,Δy, log(d),log(w),log(h), conf, cls...)

        # Initialize anchors (example for 3 scales)
        self.register_buffer(
            "anchors",
            torch.tensor(
                [
                    [0.1, 0.1, 0.1],  # Small objects (e.g., 10% of volume size)
                    [0.3, 0.3, 0.3],  # Medium objects
                    [0.5, 0.5, 0.5],  # Large objects
                ]
            ).float(),
        )

    def forward(self, x):
      
        B = x.shape[0]  # Batch size
        # Input shape: [B, C, D, H, W] = [2, 1, 32, 64, 64]
        
        # Step 1: Downsample
        out = self.downsample(x)  # [2, 1, 16, 32, 32] (stride=2)
        
        # Step 2: 1x1x1 Conv to predict anchor values
        out = self.conv(out)  # [2, 27, 16, 32, 32] (num_anchors*9=27 channels)
        
        # Step 3: Reshape to separate anchors and predictions
        out = out.view(B, self.num_anchors, 9, *out.shape[2:])  # [2, 3, 9, 16, 32, 32]
        
        # Step 4: Decode predictions (EXPLICIT SLICING)
        #  3 for delta, 3 for log_dwh,1 for conf
        delta_zxy = torch.sigmoid(out[:, :, :3, :, :, :]) * 2 - 0.5  # [2, 3, 3, 16, 32, 32]
        log_dwh = out[:, :, 3:6, :, :, :]  # [2, 3, 3, 16, 32, 32]
        conf = torch.sigmoid(out[:, :, 6, :, :])  # [2, 3, 16, 32, 32]
        # cls = torch.softmax(out[:, :, 7:, :, :, :], dim=2)  # [2, 3, 2, 16, 32, 32]
        
        return delta_zxy, log_dwh, conf



class AnchorBBox3DLoss(nn.Module):
    def __init__(self,config, pos_weight=1.0, neg_weight=0.5, lambda_reg=1.0):
        super().__init__()
        print("normal anchor bbox loss")
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.lambda_reg = lambda_reg  # Weight for regression loss
        
        self.reg_loss = nn.SmoothL1Loss(reduction='none')  # For delta_zxy and log_dwh
        self.conf_loss = nn.BCELoss(reduction='none')     # For confidence

    def forward(self, delta_zxy, log_dwh, conf_pred, gt_boxes):
        """
        Args:
            preds: Tuple of (delta_zxy, log_dwh, conf) from AnchorBBox3DHead
                - delta_zxy: (B, D,H,W, N_boxes,3) - position offsets
                - log_dwh: (B, D,H,W, N_boxes,3) - log-scale dimensions  
                - conf: (B, D,H,W, N_boxes,) - confidence scores
            gt_boxes: (B, max_objs, 6) - Each box is (cx,cy,cz,w,h,d) in normalized coords [0,1]
        """
        # delta_zxy, log_dwh, conf_pred = preds
        # print(" in loss delta_zxy shape", delta_zxy.shape, "log_dwh shape", log_dwh.shape, "conf pred", conf_pred.shape)
        
        # Fix 1: Correct variable name
        B, num_anchors, _, D, H, W = delta_zxy.shape
        
        # 1. Convert predictions to absolute box coordinates
        z_center = torch.linspace(0, 1, D, device=delta_zxy.device).view(1, 1, D, 1, 1)
        y_center = torch.linspace(0, 1, H, device=delta_zxy.device).view(1, 1, 1, H, 1)
        x_center = torch.linspace(0, 1, W, device=delta_zxy.device).view(1, 1, 1, 1, W)
        
        pred_cx = x_center + (delta_zxy[:, :, 1, ...] - 0.5)
        pred_cy = y_center + (delta_zxy[:, :, 0, ...] - 0.5)
        pred_cz = z_center + (delta_zxy[:, :, 2, ...] - 0.5)
        pred_w = torch.exp(log_dwh[:, :, 1, ...])
        pred_h = torch.exp(log_dwh[:, :, 0, ...])
        pred_d = torch.exp(log_dwh[:, :, 2, ...])
        
        pred_boxes = torch.stack([pred_cz, pred_cx, pred_cy, pred_d, pred_w, pred_h], dim=2)
        
        # 2. Match GT boxes to anchors
        with torch.no_grad():
            # Fix 2: Create separate targets for position and size
            pos_target = torch.zeros_like(delta_zxy)  # (B, num_anchors, 3, D, H, W)
            size_target = torch.zeros_like(log_dwh)   # (B, num_anchors, 3, D, H, W)
            conf_target = torch.zeros_like(conf_pred)  # (B, num_anchors, D, H, W)
            
            for b in range(B):
                # Filter empty boxes (assume padding is marked with negative values)
                valid_mask = gt_boxes[b, :, 0] >= 0
                # print("valid mask", valid_mask.shape, valid_mask.sum())
                gt_boxes_b = gt_boxes[b][valid_mask]  # (num_gt, 6)
                # print("gt boxes b", gt_boxes_b.shape)
                if len(gt_boxes_b) == 0:
                    continue
                    
                # Convert GT to grid coordinates - Fix 3: Remove double scaling
                gt_z = gt_boxes_b[:, 0]  # Already normalized [0,1]
                gt_x = gt_boxes_b[:, 1]  # Already normalized [0,1]
                gt_y = gt_boxes_b[:, 2]  # Already normalized [0,1]
                gt_d = gt_boxes_b[:, 3]  # Already normalized [0,1]
                gt_w = gt_boxes_b[:, 4]  # Already normalized [0,1]
                gt_h = gt_boxes_b[:, 5]  # Already normalized [0,1]
           
                grid_z = torch.clamp((gt_z * D).long(), 0, D-1)
                grid_x = torch.clamp((gt_x * W).long(), 0, W-1)
                grid_y = torch.clamp((gt_y * H).long(), 0, H-1)
                # print("grid_z", grid_z.shape, "grid_x", grid_x.shape, "grid_y", grid_y.shape)
                for gt_idx in range(len(gt_boxes_b)):
                    z, x, y = grid_z[gt_idx], grid_x[gt_idx], grid_y[gt_idx]
                    # print("z, x, y", z, x, y)
                    # Position targets (offsets from grid centers)
                    pos_target[b, :, 0, z, x, y] = gt_z[gt_idx] - z_center[0, 0, z, 0, 0]  # dz
                    pos_target[b, :, 1, z, x, y] = gt_x[gt_idx] - x_center[0, 0, 0, 0, x]  # dx  
                    pos_target[b, :, 2, z, x, y] = gt_y[gt_idx] - y_center[0, 0, 0, y, 0]  # dy
                    
                    # Size targets (log-scale)
                    size_target[b, :, 0, z, x, y] = torch.log(gt_h[gt_idx] + 1e-6)  # log_h
                    size_target[b, :, 1, z, x, y] = torch.log(gt_w[gt_idx] + 1e-6)  # log_w
                    size_target[b, :, 2, z, x, y] = torch.log(gt_d[gt_idx] + 1e-6)  # log_d
                    
                    conf_target[b, :, z, x, y] = 1.0
        
        reg_mask = conf_target > 0.5
        reg_mask_expanded = reg_mask.unsqueeze(2).expand(-1, -1, 3, -1, -1, -1)
        
        # Position loss
        pos_loss = self.reg_loss(delta_zxy[reg_mask_expanded], pos_target[reg_mask_expanded]).sum()
        
        # Size loss  
        size_loss = self.reg_loss(log_dwh[reg_mask_expanded], size_target[reg_mask_expanded]).sum()
        
        conf_loss = (self.conf_loss(conf_pred, conf_target) * 
                    torch.where(conf_target > 0.5, self.pos_weight, self.neg_weight)).sum()
        
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