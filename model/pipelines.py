from dotenv import load_dotenv
import os
import sys
load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
from model.bbox3d.bbox_head import AnchorBBox3DHead
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.type import dict_to_namespace
import torch

def compute_3d_volume(boxes):
    """
    Compute volume of 3D boxes.
    Args:
        boxes: (..., 6) tensor where last dim is (z,x,y,d,w,h)
    Returns:
        volumes: (...) tensor of box volumes
    """
    d, w, h = boxes[..., 3], boxes[..., 4], boxes[..., 5]
    return d * w * h

def compute_diagonal_length(boxes):
    """
    Compute diagonal length of 3D boxes.
    Args:
        boxes: (..., 6) tensor where last dim is (z,x,y,d,w,h)
    Returns:
        diags: (...) tensor of diagonal lengths
    """
    d, w, h = boxes[..., 3], boxes[..., 4], boxes[..., 5]
    return torch.sqrt(d**2 + w**2 + h**2)

def compute_3d_intersection(pred_boxes, gt_boxes):
    """
    Compute intersection volume between predicted and ground truth 3D boxes.
    Args:
        pred_boxes: (B, N, 6) - (z,x,y,d,w,h)
        gt_boxes: (B, N, 6) - (z,x,y,d,w,h)
    Returns:
        intersection: (B, N) tensor of intersection volumes
    """
    # Convert to corner format (min_z, min_x, min_y, max_z, max_x, max_y)
    def get_corners(boxes):
        z, x, y, d, w, h = boxes.unbind(-1)
        half_d, half_w, half_h = d/2, w/2, h/2
        min_z, max_z = z - half_d, z + half_d
        min_x, max_x = x - half_w, x + half_w
        min_y, max_y = y - half_h, y + half_h
        return torch.stack([min_z, min_x, min_y, max_z, max_x, max_y], dim=-1)
    
    pred_corners = get_corners(pred_boxes)  # (B, N, 6)
    gt_corners = get_corners(gt_boxes)      # (B, N, 6)
    
    # Compute intersection for each box pair
    min_z = torch.max(pred_corners[..., 0], gt_corners[..., 0])
    min_x = torch.max(pred_corners[..., 1], gt_corners[..., 1])
    min_y = torch.max(pred_corners[..., 2], gt_corners[..., 2])
    
    max_z = torch.min(pred_corners[..., 3], gt_corners[..., 3])
    max_x = torch.min(pred_corners[..., 4], gt_corners[..., 4])
    max_y = torch.min(pred_corners[..., 5], gt_corners[..., 5])
    
    # Clip to zero if no intersection
    inter_d = torch.clamp(max_z - min_z, min=0)
    inter_w = torch.clamp(max_x - min_x, min=0)
    inter_h = torch.clamp(max_y - min_y, min=0)
    
    return inter_d * inter_w * inter_h  # (B, N)

def diou_3d(pred_boxes, gt_boxes):
    """
    3D Distance-IoU Loss (DIoU)
    Args:
        pred_boxes: (B, N, 6) - (z,x,y,d,w,h)
        gt_boxes: (B, N, 6) - (z,x,y,d,w,h)
    Returns:
        diou_loss: scalar tensor
    """
    # Compute 3D IoU
    intersection = compute_3d_intersection(pred_boxes, gt_boxes)  # (B, N)
    pred_vol = compute_3d_volume(pred_boxes)  # (B, N)
    gt_vol = compute_3d_volume(gt_boxes)      # (B, N)
    union = pred_vol + gt_vol - intersection
    iou = intersection / (union + 1e-7)  # (B, N)
    
    # Compute center distance
    pred_centers = pred_boxes[..., :3]  # (B, N, 3)
    gt_centers = gt_boxes[..., :3]      # (B, N, 3)
    c_dist = torch.norm(pred_centers - gt_centers, dim=-1)  # (B, N)
    
    # Compute diagonal distance of GT boxes
    diag_dist = compute_diagonal_length(gt_boxes)  # (B, N)
    
    # DIoU = IoU - (center_dist / diagonal_dist)
    diou = iou - (c_dist / (diag_dist + 1e-7))  # (B, N)
    
    # Loss = 1 - DIoU
    return 1 - diou.mean()  # Scalar

class Projector3D(nn.Module):
    def __init__(self, vit_dim, output_channels, patches=(8, 16, 16)):
        super().__init__()
        self.patches = patches
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(vit_dim, 256, kernel_size=2, stride=2),  # 2x up
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),  # 4x up
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.out_conv = nn.Conv3d(128, output_channels, kernel_size=1)  # Channel adjust

    def forward(self, x):
        # Input: (B, N, vit_dim) → reshape to 3D
        B, N, C = x.shape
        D, H, W = self.patches
        x = x.permute(0, 2, 1).view(B, C, D, H, W)  # (B, C, D, H, W)
        x = self.up1(x)  # 2x up
        x = self.up2(x)  # 4x up
        return self.out_conv(x)  # (B, output_channels, D*4, H*4, W*4)

def decode_predictions(delta_zxy, log_dwh, conf, anchors=None, img_dims=(32, 256, 256)):
    """
    Convert raw model outputs to interpretable 3D bounding boxes
    
    Args:
        delta_zxy: (B, num_anchors, 3, D, H, W) - Center offsets in [dz, dx, dy] format
        log_dwh:   (B, num_anchors, 3, D, H, W) - Log of dimensions [log_d, log_w, log_h]
        conf:      (B, num_anchors, D, H, W)    - Confidence scores
        anchors:   (num_anchors, 3) or None      - Base anchor sizes [d, w, h]
        img_dims:  (D, H, W)                    - Original image dimensions
        
    Returns:
        pred_boxes:  List of (N, 6) tensors (one per batch) - [cx, cy, cz, w, h, d] in absolute coordinates
        pred_scores: List of (N,) tensors - Confidence scores
    """
    device = delta_zxy.device
    B, num_anchors, _, D, H, W = delta_zxy.shape
    
    # 1. Create grid of center points (normalized 0-1)
    z_grid = torch.linspace(0.5/D, 1-0.5/D, D, device=device).view(1, 1, D, 1, 1)
    y_grid = torch.linspace(0.5/H, 1-0.5/H, H, device=device).view(1, 1, 1, H, 1)
    x_grid = torch.linspace(0.5/W, 1-0.5/W, W, device=device).view(1, 1, 1, 1, W)
    
    # 2. Convert center offsets to absolute coordinates
    # Note: delta_zxy is already sigmoid-activated (from model forward)
    pred_cx = x_grid + (delta_zxy[:, :, 1, ...] - 0.5) / W  # [B, A, D, H, W]
    pred_cy = y_grid + (delta_zxy[:, :, 0, ...] - 0.5) / H  
    pred_cz = z_grid + (delta_zxy[:, :, 2, ...] - 0.5) / D
    
    # 3. Convert log dimensions to absolute sizes
    if anchors is not None:
        # Anchor-based scaling
        anchor_d = anchors[:, 0].view(1, num_anchors, 1, 1, 1)  # [1, A, 1, 1, 1]
        anchor_w = anchors[:, 1].view(1, num_anchors, 1, 1, 1)
        anchor_h = anchors[:, 2].view(1, num_anchors, 1, 1, 1)
        
        pred_d = anchor_d * torch.exp(log_dwh[:, :, 0, ...])  # [B, A, D, H, W]
        pred_w = anchor_w * torch.exp(log_dwh[:, :, 1, ...])
        pred_h = anchor_h * torch.exp(log_dwh[:, :, 2, ...])
    else:
        # Anchor-free
        pred_d = torch.exp(log_dwh[:, :, 0, ...])
        pred_w = torch.exp(log_dwh[:, :, 1, ...])
        pred_h = torch.exp(log_dwh[:, :, 2, ...])
    
    # 4. Scale to original image dimensions
    scale_z, scale_y, scale_x = img_dims
    pred_cx = pred_cx * scale_x
    pred_cy = pred_cy * scale_y
    pred_cz = pred_cz * scale_z
    pred_w = pred_w * scale_x
    pred_h = pred_h * scale_y
    pred_d = pred_d * scale_z
    
    # 5. Reshape to (B, num_anchors*D*H*W, 6)
    pred_boxes = torch.stack([pred_cx, pred_cy, pred_cz, pred_w, pred_h, pred_d], dim=2)
    pred_boxes = pred_boxes.permute(0, 1, 3, 4, 5, 2).reshape(B, -1, 6)
    
    # 6. Flatten confidence scores
    pred_scores = torch.sigmoid(conf).reshape(B, -1)
    
    # Convert to per-sample lists (useful for variable numbers of predictions)
    return [pred_boxes[i] for i in range(B)], [pred_scores[i] for i in range(B)] 

class ViT3DDetector(nn.Module):
    def __init__(self, vit, projector, bbox_head):
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.bbox_head = bbox_head

    def forward(self, x):
        # ViT backbone
        vit_features = self.vit(x)  # (B, N, C), [skip1, skip2, ...]
        print("ViT features shape:", vit_features.shape)
        # torch.Size([2, 2048, 768]
        # Projector (Decoder)
        volume = self.projector(vit_features)  # (B, C, D, H, W)
        print("Projected volume shape:", volume.shape)
        # BBox Head
        Δzxy, log_dwh, conf, cls = self.bbox_head(volume)
        return Δzxy, log_dwh, conf, cls
if __name__=="__main__":
    from model.Encoder.vit import ViT3DTower 
    vit=ViT3DTower()
    projector=Projector3D(vit_dim=768, output_channels=1, patches=(8, 16, 16)) 
    bbox_head=AnchorBBox3DHead(in_channels=1)
    model = ViT3DDetector(vit, projector, bbox_head)
    x = torch.randn(2, 1, 32, 256, 256)  # Example input
    delta_zxy, log_dwh, conf, cls = model(x)
    print("Δzxy:", delta_zxy.shape, delta_zxy.min(), delta_zxy.max())
    print("log_dwh:", log_dwh.shape, log_dwh.min(), log_dwh.max())
    print("conf:", conf.shape, conf.min(), conf.max())
    print("cls:", cls.shape, cls.min(), cls.max())
    

    # Δzxy: torch.Size([2, 3, 3, 16, 32, 32]) tensor(0.2623, grad_fn=<MinBackward1>) tensor(0.9434, grad_fn=<MaxBackward1>)
    # log_dwh: torch.Size([2, 3, 3, 16, 32, 32]) tensor(-1.3723, grad_fn=<MinBackward1>) tensor(1.2363, grad_fn=<MaxBackward1>)
    # conf: torch.Size([2, 3, 16, 32, 32]) tensor(0.2678, grad_fn=<MinBackward1>) tensor(0.7361, grad_fn=<MaxBackward1>)
    # cls: torch.Size([2, 3, 2, 16, 32, 32]) tensor(0.1732, grad_fn=<MinBackward1>) tensor(0.8268, grad_fn=<MaxBackward1>)