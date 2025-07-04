import torch 
from torch import nn
from typing import Optional
import torch.nn.functional as F
from types import SimpleNamespace

import sys 
from dotenv import load_dotenv
import os
load_dotenv()
ROOT=os.getenv("ROOT")
sys.path.append(ROOT)
import torch
from scipy.optimize import linear_sum_assignment

class BBox3DPredictor(nn.Module):
    """Handles 3D bounding box prediction"""

    def __init__(self):
        super().__init__()
        config={
            "bbox3d_module": True,
            "mm_hidden_size": 512,  # Example size, adjust as needed
            "bbox3d_head_type": "default",  # Placeholder for bbox3d head type
            "bbox3d_projector_type": "default",  # Placeholder for bbox3d
        }
        config= SimpleNamespace(**config)
        self.config = config
        self.bbox3d_head = None
        self.bbox3d_projector = None
        self.enabled = False
        self.loss_calculator = None

        if config.bbox3d_module:
            try:
                self._build_components()
            except Exception as e:
                print(f"Warning: Failed to build bbox3d components: {e}")

    def _build_components(self):
        """Build bbox3d components"""
        self.bbox3d_head = self._build_bbox3d_head()
        self.bbox3d_projector = nn.Sequential(
            nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size),
            nn.Dropout(0.1),
        )
        self.loss_calculator = self._create_loss_calculator()
        self.enabled = True

    def _build_bbox3d_head(self):
        """Build bbox3d head - implement based on your builder"""
        try:
            from model.bbox3d.bbox_head import build_bbox3d_module

            return build_bbox3d_module()
        except ImportError as e:
            raise ImportError(f"Failed to import bbox3d builder: {e}")

    def _create_loss_calculator(self):
        """Create loss calculation module"""
        try:
            from model.loss import L1Loss, IoU3DLoss

            class BBox3DLossCalculator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1_loss = L1Loss()
                    self.iou3d_loss = IoU3DLoss()

                def compute_loss(self, predictions, targets, masks=None):
                    """Compute bbox3d loss with proper masking"""
                    if masks is not None:
                        # Apply mask to compute loss only on valid boxes
                        valid_mask = masks.unsqueeze(-1).float()
                        masked_pred = predictions * valid_mask
                        masked_target = targets * valid_mask

                        l1_loss = self.l1_loss(masked_pred, masked_target)
                        iou_loss = self.iou3d_loss(masked_pred, masked_target)

                        # Normalize by number of valid boxes
                        num_valid = masks.sum().clamp(min=1)
                        return (l1_loss + iou_loss) / num_valid
                    else:
                        return self.l1_loss(predictions, targets) + self.iou3d_loss(
                            predictions, targets
                        )

            return BBox3DLossCalculator()
        except ImportError:
            return nn.MSELoss()

    def extract_bbox_features(
        self, hidden_states: torch.Tensor, bbox_token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract bbox features from hidden states"""
        bbox_prompts = []

        for i in range(bbox_token_mask.shape[0]):
            token_count = torch.sum(bbox_token_mask[i])

            if token_count == 1:
                bbox_token = hidden_states[i][bbox_token_mask[i]]
                bbox_prompt = self.bbox3d_projector(bbox_token)
            elif token_count > 1:
                bbox_tokens = hidden_states[i][bbox_token_mask[i]]
                bbox_token = torch.mean(bbox_tokens, dim=0, keepdim=True)
                bbox_prompt = self.bbox3d_projector(bbox_token)
            else:
                bbox_prompt = torch.zeros(
                    [1, self.config.mm_hidden_size],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            bbox_prompts.append(bbox_prompt)

        return torch.cat(bbox_prompts, dim=0)

    def predict_bboxes(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict 3D bounding boxes"""
        if not self.enabled:
            return None

        try:
            # Pool features
            vision_pooled = vision_features.mean(dim=1)  # [B, D_v]
            text_pooled = text_features.mean(dim=1)  # [B, D_t]

            # print("vision_pooled",vision_pooled.shape)
            # print("text_pooled",text_pooled.shape)
            # Combine features
            combined = torch.cat([vision_pooled, text_pooled], dim=-1)
            # print("combined",combined.shape)
            # Predict bboxes
            return self.bbox3d_head(combined)
        except Exception as e:
            raise Exception(f"Warning: Failed to predict bboxes: {e}")

    def compute_bbox_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute bbox prediction loss with proper shape handling"""
        bbox_preds = predictions["filtered_bbox_pred"]
        conf_preds = predictions["filtered_conf_pred"]
        # total_loss = torch.tensor(0.0, device=bbox_preds[0].device)
        losses = []
        for b in range(len(bbox_preds)):
            bbox_pred = bbox_preds[b]        # [num_preds, 6], in center format
            conf_pred = conf_preds[b]        # [num_preds]
            mask = masks[b]                  # [max_num_gt]
            targets_b = targets[b]           # [max_num_gt, 6], center format

            #  Get most confident prediction
            conf_idx = conf_pred.argmax()
            top_pred = bbox_pred[conf_idx]   # [6] — in center format

            #  Filter valid GTs
            valid_gt_boxes = targets_b[mask]  # [num_valid_gt, 6], center format

            if len(valid_gt_boxes) == 0:
                continue  # skip if no valid GTs

            # Convert both to min-max for IoU
            top_pred_minmax = self.bbox3d_head.convert_model_to_gt_format(top_pred.unsqueeze(0))       # [1, 6]
            valid_gt_minmax = self.bbox3d_head.convert_model_to_gt_format(valid_gt_boxes)              # [num_valid_gt, 6]

            #  Match to best GT via IoU
            ious = compute_3d_iou_matrix(top_pred_minmax, valid_gt_minmax)  # [1, num_valid_gt]
            best_gt_idx = ious[0].argmax()
            best_gt = valid_gt_boxes[best_gt_idx]  # [6], still center format

            #  Compute regression loss (center format vs center format)
            loss = F.smooth_l1_loss(top_pred.unsqueeze(0), best_gt.unsqueeze(0))
            total_loss += loss

        return torch.stack(losses).sum()


def compute_3d_iou_matrix(pred_boxes, gt_boxes):
    """
    Compute IoU matrix between predicted and ground truth 3D boxes
    Args:
        pred_boxes: [N, 6] in format [x_min, y_min, z_min, x_max, y_max, z_max]
        gt_boxes:   [M, 6] in same format
    Returns:
        iou_matrix: [N, M] IoU values
    """
    N = pred_boxes.size(0)
    M = gt_boxes.size(0)

    iou_matrix = torch.zeros(N, M, device=pred_boxes.device)

    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = box3d_iou_single(pred_boxes[i], gt_boxes[j])

    return iou_matrix


def box3d_iou_single(box1, box2):
    """
    Compute IoU between two axis-aligned 3D boxes
    Args:
        box1, box2: [6] tensor [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns:
        iou: scalar IoU
    """
    # Intersection box
    inter_min = torch.max(box1[:3], box2[:3])
    inter_max = torch.min(box1[3:], box2[3:])
    inter_dim = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_dim.prod()

    # Volumes
    vol1 = (box1[3:] - box1[:3]).prod()
    vol2 = (box2[3:] - box2[:3]).prod()

    union_vol = vol1 + vol2 - inter_vol + 1e-8
    iou = inter_vol / union_vol
    return iou


def match_boxes(pred_boxes, gt_boxes):
    """
    Match predicted boxes to GT boxes using Hungarian matching on IoU
    Args:
        pred_boxes: [N, 6] tensor
        gt_boxes: [M, 6] tensor
    Returns:
        matched_pred_indices: list of indices into pred_boxes
        matched_gt_indices: list of indices into gt_boxes
    """
    iou_matrix = compute_3d_iou_matrix(pred_boxes, gt_boxes)  # [N, M]
    cost_matrix = 1.0 - iou_matrix.cpu().numpy()  # Hungarian minimizes cost

    pred_inds, gt_inds = linear_sum_assignment(cost_matrix)

    return pred_inds, gt_inds, iou_matrix[pred_inds, gt_inds]  # optional: return matched IoUs



if __name__=="__main__":
    #  assume bbox in format [center_x, center_y, center_z, width, height, length]
    builder = BBox3DPredictor()
    target=torch.randn(2,3,6)
    vision_features=torch.randn(2, 256, 3072)
    text_features=torch.randn(2, 765, 3072)
    predictions=builder.predict_bboxes(vision_features,text_features)
    masks = torch.ones(2, 3, dtype=torch.bool)
    bbox_loss = builder.compute_bbox_loss(
            predictions, target, masks
        )