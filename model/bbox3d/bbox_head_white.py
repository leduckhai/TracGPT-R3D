import torch
from torch import nn
from typing import Optional
import sys
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
import torch
import torch.nn.functional as F

class AnchorBBox3DHeadV2(nn.Module):
    def __init__(self,config,num_anchors=1,patch_grid=[4,4,4],embed_dim=768,in_channels=64):
         # Stronger center prediction head
        super().__init__()
        self.num_anchors = num_anchors
        self.patch_grid = patch_grid
        self.center_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_anchors * 6)  
        )
        
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        B = x.shape[0]
        
        center_logits = self.center_head(x)  # [B, N, 1]
        center_pred = torch.sigmoid(center_logits / 0.5).squeeze(-1)  # Sharpened
        
        bbox_params = self.bbox_head(x).view(B, *self.patch_grid, self.num_anchors, 6)
        
        delta_xyz = torch.tanh(bbox_params[..., :3])  
        log_dwh = bbox_params[..., 3:6]  
        # conf_logits = bbox_params[..., 6]  # Confidence scores
        
        return center_pred.view(B, *self.patch_grid), delta_xyz, log_dwh
       
    
class AnchorBBox3DLossV2(nn.Module):
    def __init__(self,  pos_weight=1.0, neg_weight=0.5, lambda_reg=1.0):
        super().__init__()
        print("V2 anchor bbox loss")
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.lambda_reg = lambda_reg
        
        self.reg_loss = nn.SmoothL1Loss(reduction='sum')
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none')  
        self.anchors=[]
        self.num_anchors = 1
        self.focal_loss = FocalLossSigmoid(gamma=2.0, alpha=0.75)
        # self.focal_loss = StableFocalLoss(gamma=2.0, alpha=0.45)
    def forward(self,center_pred, delta_zxy, log_dwh, gt_boxes, masks):
        B, D, H, W, num_anchors, _ = delta_zxy.shape
        device = delta_zxy.device
        
        # Grid centers
        z_grid = torch.linspace(0.5/D, 1-0.5/D, D, device=device).view(1,D,1,1,1)
        y_grid = torch.linspace(0.5/H, 1-0.5/H, H, device=device).view(1,1,H,1,1)
        x_grid = torch.linspace(0.5/W, 1-0.5/W, W, device=device).view(1,1,1,W,1)
        
        # Initialize targets
        pos_target = torch.zeros_like(delta_zxy)
        size_target = torch.zeros_like(log_dwh)
        gt_center = torch.zeros_like(center_pred)
        
        total_gt_boxes = 0
        for b in range(B):
            gt = gt_boxes[b][masks[b]]
            # print("gt",gt )
            total_gt_boxes += len(gt)
            if len(gt) == 0:
                print("no gt")
                continue
                
            grid_z = (gt[:,0] * D).clamp(0, D-1).long()
            grid_y = (gt[:,1] * H).clamp(0, H-1).long()
            grid_x = (gt[:,2] * W).clamp(0, W-1).long()
            best_ious=[]
            for gt_idx in range(len(gt)):
                z, y, x = grid_z[gt_idx], grid_y[gt_idx], grid_x[gt_idx]
                pred_vol=   torch.exp(log_dwh[b,z,y,x,:] + 1e-6).squeeze()
                # print("pred vol", pred_vol.shape,torch.exp(log_dwh[b,z,y,x,:]), gt[gt_idx,3:6]  )
                gt_box = gt[gt_idx, 3:6].unsqueeze(0)
                ious = calculate_3d_iou(gt_box, pred_vol)
               
                best_iou, best_box = ious.max(dim=1)
                if best_box >= num_anchors:  
                    print("invalid best box", best_box)
                    continue
                best_ious.append(best_iou)
                    
                gt_center[b,z,y,x] = 1
                pos_target[b,z,y,x,best_box,0] = gt[gt_idx,0] - z_grid[0,z,0,0,0]
                pos_target[b,z,y,x,best_box,1] = gt[gt_idx,1] - y_grid[0,0,y,0,0]
                pos_target[b,z,y,x,best_box,2] = gt[gt_idx,2] - x_grid[0,0,0,x,0]
                
                size_target[b,z,y,x,best_box] = torch.log(gt[gt_idx,3:6] + 1e-6)
            print("best ious mean", torch.tensor(best_ious).mean().item())
        total_sum = 0
        
        center_loss = self.focal_loss(center_pred, gt_center)
        
        pos_mask = gt_center.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,-1,  num_anchors,3).bool() 
        print("pos mask",pos_mask.sum())
        pos_loss = self.reg_loss(delta_zxy[pos_mask], pos_target[pos_mask])
        size_loss = self.reg_loss(log_dwh[pos_mask], size_target[pos_mask])
        
        num_pos = max(1.0,gt_center.sum())
        total_loss = (center_loss + pos_loss + size_loss) / num_pos
        
        return {
            'total_loss': total_loss,
            'pos_loss': pos_loss/num_pos,
            'size_loss': size_loss/num_pos,
            'center_loss': center_loss/num_pos
        }

def calculate_3d_iou(box1, box2):
    """
    Calculate 3D IoU between two sets of boxes
    
    Args:
        box1: (N, 3) tensor of (width, height, depth)
        box2: (M, 3) tensor of (width, height, depth)
        
    Returns:
        iou: (N, M) tensor of IoU values
    """
    # Expand dimensions for broadcasting
    box1 = box1.unsqueeze(1)  # (N, 1, 3)
    box2 = box2.unsqueeze(0)  # (1, M, 3)
    
    # Calculate volumes
    vol1 = box1[..., 0] * box1[..., 1] * box1[..., 2]  # (N, 1)
    vol2 = box2[..., 0] * box2[..., 1] * box2[..., 2]  # (1, M)
    
    # Find intersection dimensions
    min_w = torch.min(box1[..., 0], box2[..., 0])  # (N, M)
    min_h = torch.min(box1[..., 1], box2[..., 1])  # (N, M)
    min_d = torch.min(box1[..., 2], box2[..., 2])  # (N, M)
    
    # Calculate intersection volume
    intersection = min_w * min_h * min_d
    intersection = torch.clamp(intersection, min=0)
    
    # Calculate union volume
    union = vol1 + vol2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-6)
    
    return iou

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Original implementation from 
    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples. Default = 2.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation: https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class FocalLossSigmoid(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha       # Weight for foreground (higher = focus more on foreground)
        self.gamma = gamma       # Penalty exponent (higher = harder on low-confidence predictions)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted foreground probabilities (after sigmoid) [B, D, H, W]
            targets: Ground truth (1=foreground, 0=background) [B, D, H, W]
        """
        targets = targets.float()

        # Binary Cross-Entropy (since inputs are already sigmoided)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Focal modulation: Focus on low-confidence foreground predictions
        p_t = inputs * targets + (1 - inputs) * (1 - targets)  # p if foreground, 1-p if background
        focal_loss = bce_loss * ((1 - p_t) ** self.gamma)

        # Apply alpha weighting (higher alpha = focus more on foreground)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class StableFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, topk_ratio=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.topk_ratio = topk_ratio

    def forward(self, inputs, targets):
        # 1. Select Top-K predictions
        k = int(self.topk_ratio * inputs.numel())
        topk_values, topk_indices = torch.topk(inputs.flatten(), k=k)
        mask = torch.zeros_like(inputs)
        mask.view(-1)[topk_indices] = 1

        # 2. Compute focal loss only on Top-K
        p = torch.sigmoid(inputs) * mask
        bce_loss = F.binary_cross_entropy(p, targets * mask, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        return focal_loss.sum()