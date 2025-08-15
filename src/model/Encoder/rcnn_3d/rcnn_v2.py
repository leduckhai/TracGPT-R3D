import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
import monai
from monai.networks.nets import ResNet
import numpy as np
from typing import List, Dict
class RoIAlign3D(nn.Module):
    def __init__(self, output_size=7, spatial_scale=1.0):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        """
        Args:
            features (Tensor): (B, C, D, H, W)
            rois (Tensor): (N, 7) [batch_idx, z1, y1, x1, z2, y2, x2]
        Returns:
            Pooled features (Tensor): (N, C, output_size, output_size, output_size)
        """
        batch_size, num_channels = features.shape[:2]
        num_rois = rois.shape[0]
        output = torch.zeros(
            (num_rois, num_channels, 
             self.output_size, self.output_size, self.output_size),
            device=features.device
        )

        for roi_idx in range(num_rois):
            batch_idx, z1, y1, x1, z2, y2, x2 = rois[roi_idx]
            batch_idx = int(batch_idx)
            
            # Convert ROI coordinates to feature map scale
            z1 = int(z1 * self.spatial_scale)
            y1 = int(y1 * self.spatial_scale)
            x1 = int(x1 * self.spatial_scale)
            z2 = int(z2 * self.spatial_scale) + 1  # Make end-exclusive
            y2 = int(y2 * self.spatial_scale) + 1
            x2 = int(x2 * self.spatial_scale) + 1
            
            # Extract ROI
            roi_features = features[batch_idx, :, z1:z2, y1:y2, x1:x2]
            
            if roi_features.numel() == 0:
                continue
                
            # Adaptive 3D pooling
            output[roi_idx] = F.adaptive_avg_pool3d(
                roi_features, 
                self.output_size
            )
            
        return output
    
class FasterRCNN3D(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet18'):
        super().__init__()
        
        # 1. Backbone Network (3D)
        self.backbone = ResNet(
            block='basic',
            layers=[2, 2, 2, 2],  # resnet18
            spatial_dims=3,
            n_input_channels=1,
            pretrained=False
        )
        
        # 2. Region Proposal Network (RPN)
        self.rpn_conv = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.rpn_cls = nn.Conv3d(512, 9 * 1, kernel_size=1)  # 9 anchors per position, binary classification
        self.rpn_reg = nn.Conv3d(512, 9 * 6, kernel_size=1)  # 6 coords per anchor (dz,dy,dx,dd,dh,dw)
        
        # 3. ROI Pooling (Custom 3D)
        self.roi_pool = RoIAlign3D(output_size=7, spatial_scale=1/16.0)
        
        # 4. Detection Head
        self.head = nn.Sequential(
            nn.Linear(512*7*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes * 7)  # 4 box coords + 3 class scores
        )
        
        # Anchor parameters
        self.anchor_sizes = [(10,10,10), (20,20,20), (30,30,30)]  # in mm
        self.anchor_ratios = [0.5, 1, 2]

    def forward(self, x: torch.Tensor, targets: List[Dict[str, torch.Tensor]] = None):
        # x: (B, C, D, H, W)
        features = self.backbone(x)  # (B, 512, D/16, H/16, W/16)
        
        # RPN
        rpn_feat = F.relu(self.rpn_conv(features))
        rpn_logits = self.rpn_cls(rpn_feat)  # (B, 9, D', H', W')
        rpn_reg = self.rpn_reg(rpn_feat)     # (B, 54, D', H', W')
        
        if self.training:
            return self._train(features, rpn_logits, rpn_reg, targets)
        else:
            return self._inference(features, rpn_logits, rpn_reg)
    
    def _train(self, features, rpn_logits, rpn_reg, targets):
        """Training forward pass with losses"""
        # 1. Generate anchors
        anchors = self._generate_anchors(features.shape[2:])  # (D',H',W')
        
        # 2. Calculate RPN losses
        rpn_loss = self._calc_rpn_loss(rpn_logits, rpn_reg, anchors, targets)
        
        # 3. Sample proposals
        proposals = self._get_proposals(rpn_reg, anchors)
        
        # 4. ROI Pooling
        pooled = self.roi_pool(features, proposals)
        
        # 5. Calculate detection losses
        cls_logits, box_preds = self.head(pooled.flatten(1))
        det_loss = self._calc_detection_loss(cls_logits, box_preds, proposals, targets)
        
        return {
            'rpn_cls_loss': rpn_loss['cls'],
            'rpn_reg_loss': rpn_loss['reg'],
            'det_cls_loss': det_loss['cls'],
            'det_reg_loss': det_loss['reg']
        }
    
    def _inference(self, features, rpn_logits, rpn_reg):
        """Inference forward pass"""
        # 1. Generate anchors
        anchors = self._generate_anchors(features.shape[2:])
        
        # 2. Decode proposals
        proposals = self._decode_boxes(rpn_reg, anchors)
        
        # 3. Apply NMS
        scores = rpn_logits.sigmoid()
        keep = self._nms_3d(proposals, scores, iou_threshold=0.3)
        proposals = proposals[keep]
        scores = scores[keep]
        
        # 4. ROI Pooling
        pooled = self.roi_pool(features, proposals)
        
        # 5. Final predictions
        cls_logits, box_preds = self.head(pooled.flatten(1))
        
        return {
            'boxes': proposals,
            'scores': cls_logits.softmax(dim=1)[:, 1],  # Class 1 probability
            'labels': torch.ones(len(proposals))  # Single-class
        }
    
    def _generate_anchors(self, feature_shape):
        """Generate 3D anchor boxes"""
        anchors = []
        for z in range(feature_shape[0]):
            for y in range(feature_shape[1]):
                for x in range(feature_shape[2]):
                    center = (z*16+8, y*16+8, x*16+8)  # Assuming stride=16
                    for size in self.anchor_sizes:
                        for ratio in self.anchor_ratios:
                            # Convert ratio to 3D dimensions
                            d = size[0] * ratio
                            h = size[1] / ratio
                            w = size[2]  # Keep original depth
                            anchors.append([
                                center[0]-d/2, center[1]-h/2, center[2]-w/2,
                                center[0]+d/2, center[1]+h/2, center[2]+w/2
                            ])
        return torch.tensor(anchors, device=self.device)
    
    def _nms_3d(self, boxes, scores, iou_threshold):
        """3D Non-Maximum Suppression"""
        # Sort by score
        idxs = scores.argsort(descending=True)
        keep = []
        
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            
            # Calculate 3D IoU with remaining boxes
            ious = self._box3d_iou(boxes[i], boxes[idxs[1:]])
            
            # Remove boxes with IoU > threshold
            idxs = idxs[1:][ious <= iou_threshold]
            
        return torch.tensor(keep, device=boxes.device)
    
    def _box3d_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate 3D IoU between two sets of boxes.
        Args:
            box1: (N, 6) tensor of (z1,y1,x1,z2,y2,x2)
            box2: (M, 6) tensor
        Returns:
            iou: (N, M) tensor
        """
        # Expand dimensions for broadcasting
        box1 = box1.unsqueeze(1)  # (N,1,6)
        box2 = box2.unsqueeze(0)  # (1,M,6)
        
        # Calculate intersection coordinates
        max_zyx = torch.min(box1[..., 3:], box2[..., 3:])
        min_zyx = torch.max(box1[..., :3], box2[..., :3])
        
        # Intersection volumes
        inter_zyx = torch.clamp(max_zyx - min_zyx, min=0)
        inter_vol = inter_zyx[..., 0] * inter_zyx[..., 1] * inter_zyx[..., 2]
        
        # Union volumes
        vol1 = (box1[..., 3]-box1[..., 0]) * (box1[..., 4]-box1[..., 1]) * (box1[..., 5]-box1[..., 2])
        vol2 = (box2[..., 3]-box2[..., 0]) * (box2[..., 4]-box2[..., 1]) * (box2[..., 5]-box2[..., 2])
        
        return inter_vol / (vol1 + vol2 - inter_vol + 1e-6)


    def _calc_rpn_loss(self, logits: torch.Tensor, reg: torch.Tensor, 
                    anchors: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Calculate RPN classification and regression losses.
        Args:
            logits: (B, A, D, H, W) - RPN class logits
            reg: (B, A*6, D, H, W) - RPN box regression
            anchors: (N, 6) - Generated anchor boxes
            targets: List of dicts with 'boxes' and 'labels'
        Returns:
            Dictionary with 'cls' and 'reg' losses
        """
        batch_size = logits.shape[0]
        device = logits.device
        total_cls_loss = 0
        total_reg_loss = 0
        
        # Reshape predictions
        logits = logits.permute(0,2,3,4,1).reshape(-1, 1)  # (B*D*H*W*A, 1)
        reg = reg.permute(0,2,3,4,1).reshape(-1, 6)        # (B*D*H*W*A, 6)
        
        for b in range(batch_size):
            gt_boxes = targets[b]['boxes']
            
            if len(gt_boxes) == 0:
                continue
                
            # 1. Match anchors to GT boxes
            ious = self._box3d_iou(anchors, gt_boxes)
            max_ious, gt_ids = ious.max(dim=1)
            
            # 2. Assign labels (1=positive, 0=negative, -1=ignore)
            labels = torch.full((len(anchors),), -1, dtype=torch.float32, device=device)
            labels[max_ious < 0.3] = 0
            labels[max_ious >= 0.7] = 1
            
            # 3. Sample balanced batches
            pos_idx = torch.where(labels == 1)[0]
            neg_idx = torch.where(labels == 0)[0]
            
            num_pos = min(128, len(pos_idx))
            num_neg = min(256, len(neg_idx))
            
            if num_pos > 0:
                pos_idx = pos_idx[torch.randperm(len(pos_idx))[:num_pos]]
            if num_neg > 0:
                neg_idx = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]
            
            # 4. Calculate classification loss (binary CE)
            valid_idx = torch.cat([pos_idx, neg_idx])
            cls_loss = F.binary_cross_entropy_with_logits(
                logits[valid_idx], 
                labels[valid_idx].unsqueeze(1)
            )
            
            # 5. Calculate regression loss (smooth L1)
            if num_pos > 0:
                pos_anchors = anchors[pos_idx]
                pos_reg = reg[pos_idx]
                gt_pos = gt_boxes[gt_ids[pos_idx]]
                
                # Calculate box deltas
                gt_deltas = self._box_to_delta(pos_anchors, gt_pos)
                reg_loss = F.smooth_l1_loss(pos_reg, gt_deltas, reduction='sum') / num_pos
            else:
                reg_loss = torch.tensor(0., device=device)
                
            total_cls_loss += cls_loss
            total_reg_loss += reg_loss
        
        return {
            'cls': total_cls_loss / batch_size,
            'reg': total_reg_loss / batch_size
        }

    def _box_to_delta(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert GT boxes to delta format relative to anchors.
        Args:
            anchors: (N, 6)
            gt_boxes: (N, 6)
        Returns:
            deltas: (N, 6) [dz, dy, dx, dd, dh, dw]
        """
        anchor_z = (anchors[:, 0] + anchors[:, 3]) / 2
        anchor_y = (anchors[:, 1] + anchors[:, 4]) / 2
        anchor_x = (anchors[:, 2] + anchors[:, 5]) / 2
        anchor_d = anchors[:, 3] - anchors[:, 0]
        anchor_h = anchors[:, 4] - anchors[:, 1]
        anchor_w = anchors[:, 5] - anchors[:, 2]
        
        gt_z = (gt_boxes[:, 0] + gt_boxes[:, 3]) / 2
        gt_y = (gt_boxes[:, 1] + gt_boxes[:, 4]) / 2
        gt_x = (gt_boxes[:, 2] + gt_boxes[:, 5]) / 2
        gt_d = gt_boxes[:, 3] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 4] - gt_boxes[:, 1]
        gt_w = gt_boxes[:, 5] - gt_boxes[:, 2]
        
        eps = 1e-6
        dz = (gt_z - anchor_z) / (anchor_d + eps)
        dy = (gt_y - anchor_y) / (anchor_h + eps)
        dx = (gt_x - anchor_x) / (anchor_w + eps)
        dd = torch.log(gt_d / (anchor_d + eps))
        dh = torch.log(gt_h / (anchor_h + eps))
        dw = torch.log(gt_w / (anchor_w + eps))
        
        return torch.stack([dz, dy, dx, dd, dh, dw], dim=1)

    def _delta_to_box(self, anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        Convert deltas back to absolute box coordinates.
        Args:
            anchors: (N, 6)
            deltas: (N, 6)
        Returns:
            boxes: (N, 6)
        """
        anchor_z = (anchors[:, 0] + anchors[:, 3]) / 2
        anchor_y = (anchors[:, 1] + anchors[:, 4]) / 2
        anchor_x = (anchors[:, 2] + anchors[:, 5]) / 2
        anchor_d = anchors[:, 3] - anchors[:, 0]
        anchor_h = anchors[:, 4] - anchors[:, 1]
        anchor_w = anchors[:, 5] - anchors[:, 2]
        
        dz = deltas[:, 0]
        dy = deltas[:, 1]
        dx = deltas[:, 2]
        dd = deltas[:, 3]
        dh = deltas[:, 4]
        dw = deltas[:, 5]
        
        pred_z = dz * anchor_d + anchor_z
        pred_y = dy * anchor_h + anchor_y
        pred_x = dx * anchor_w + anchor_x
        pred_d = torch.exp(dd) * anchor_d
        pred_h = torch.exp(dh) * anchor_h
        pred_w = torch.exp(dw) * anchor_w
        
        return torch.stack([
            pred_z - pred_d/2, pred_y - pred_h/2, pred_x - pred_w/2,
            pred_z + pred_d/2, pred_y + pred_h/2, pred_x + pred_w/2
        ], dim=1)

    def _get_proposals(self, reg_pred: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """
        Generate proposals from RPN outputs.
        Args:
            reg_pred: (B, A*6, D, H, W) - RPN regression outputs
            anchors: (N, 6) - Anchor boxes
        Returns:
            proposals: (N, 6) - Refined boxes
        """
        # Reshape predictions
        reg_pred = reg_pred.permute(0,2,3,4,1).reshape(-1, 6)  # (N,6)
        
        # Decode boxes
        proposals = self._delta_to_box(anchors, reg_pred)
        
        # Clip to image boundaries
        proposals = torch.clamp(proposals, min=0, max=512)  # Assuming 512x512x512 volume
        
        return proposals