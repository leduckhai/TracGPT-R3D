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
from model.bbox3d.bbox_head import BBox3DHead
from model.bbox3d.helper import hungarian_iou_matching,convert_model_to_gt_format

class BBox3DPredictor(nn.Module):
    """Handles 3D bounding box prediction"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.bbox3d_head = None
        self.bbox3d_projector = None
        self.enabled = False
        self.loss_calculator = None

        self._build_components()

    def _build_components(self):
        """Build bbox3d components"""
        self.bbox3d_head = BBox3DHead(self.config.bbox_head)
        self.bbox3d_projector = nn.Sequential(
            nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size),
            nn.Dropout(0.1),
        )
        # self.loss_calculator = self._create_loss_calculator()
        self.enabled = True

    # def _create_loss_calculator(self):
    #     """Create loss calculation module"""
    #     try:
    #         from model.loss import L1Loss, IoU3DLoss

    #         class BBox3DLossCalculator(nn.Module):
    #             def __init__(self):
    #                 super().__init__()
    #                 self.l1_loss = L1Loss()
    #                 self.iou3d_loss = IoU3DLoss()

    #             def compute_loss(self, predictions, targets, masks=None):
    #                 """Compute bbox3d loss with proper masking"""
    #                 if masks is not None:
    #                     # Apply mask to compute loss only on valid boxes
    #                     valid_mask = masks.unsqueeze(-1).float()
    #                     masked_pred = predictions * valid_mask
    #                     masked_target = targets * valid_mask

    #                     l1_loss = self.l1_loss(masked_pred, masked_target)
    #                     iou_loss = self.iou3d_loss(masked_pred, masked_target)

    #                     # Normalize by number of valid boxes
    #                     num_valid = masks.sum().clamp(min=1)
    #                     return (l1_loss + iou_loss) / num_valid
    #                 else:
    #                     return self.l1_loss(predictions, targets) + self.iou3d_loss(
    #                         predictions, targets
    #                     )

    #         return BBox3DLossCalculator()
    #     except ImportError:
    #         return nn.MSELoss()

    # def extract_bbox_features(
    #     self, hidden_states: torch.Tensor, bbox_token_mask: torch.Tensor
    # ) -> torch.Tensor:
    #     """Extract bbox features from hidden states"""
    #     bbox_prompts = []

    #     for i in range(bbox_token_mask.shape[0]):
    #         token_count = torch.sum(bbox_token_mask[i])

    #         if token_count == 1:
    #             bbox_token = hidden_states[i][bbox_token_mask[i]]
    #             bbox_prompt = self.bbox3d_projector(bbox_token)
    #         elif token_count > 1:
    #             bbox_tokens = hidden_states[i][bbox_token_mask[i]]
    #             bbox_token = torch.mean(bbox_tokens, dim=0, keepdim=True)
    #             bbox_prompt = self.bbox3d_projector(bbox_token)
    #         else:
    #             bbox_prompt = torch.zeros(
    #                 [1, self.config.mm_hidden_size],
    #                 dtype=hidden_states.dtype,
    #                 device=hidden_states.device,
    #             )
    #         bbox_prompts.append(bbox_prompt)

    #     return torch.cat(bbox_prompts, dim=0)

    def predict_bboxes(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict 3D bounding boxes"""
        if not self.enabled:
            return None

        try:
            vision_pooled = vision_features.mean(dim=1)  # [B, D_v]
            text_pooled = text_features.mean(dim=1)  # [B, D_t]

            combined = torch.cat([vision_pooled, text_pooled], dim=-1)
            return self.bbox3d_head(combined)
        except Exception as e:
            raise Exception(f"Warning: Failed to predict bboxes: {e}")

    def compute_bbox_loss(
        self,
        bbox_preds: torch.Tensor,
        targets: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optimized bbox loss computation with vectorized operations"""
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(len(bbox_preds)):
            bbox_pred = bbox_preds[b]  # [num_preds, 6]
            mask = masks[b]  # [max_num_gt]
            target = targets[b]  # [max_num_gt, 6]
            gt_boxes = target[mask]  # [num_valid_gt, 6]

            if len(gt_boxes) == 0:
                continue
                
            # Convert format if needed
            pred_boxes_minmax = convert_model_to_gt_format(bbox_pred, normalize_coords=False)
            # gt_boxes_minmax = convert_model_to_gt_format(gt_boxes, normalize_coords=False)
            
            # Vectorized matching and loss computation
            matches = hungarian_iou_matching(pred_boxes_minmax, gt_boxes)
            if matches:
                pred_indices = [m[0] for m in matches]
                gt_indices = [m[1] for m in matches]
                batch_loss = F.smooth_l1_loss(
                    bbox_pred[pred_indices],
                    gt_boxes[gt_indices],
                    reduction='sum'  # Preserve magnitude
                )
                total_loss += batch_loss
                valid_batches += 1

        # Normalize by number of valid batches and matches
        return total_loss / max(1, valid_batches)

    
if __name__ == "__main__":
    import yaml
    from utils.type import dict_to_namespace

    config_path = "config/llama.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)

    #  assume bbox in format [center_x, center_y, center_z, width, height, length]
    builder = BBox3DPredictor(config.tiny_llama.bbox_predictor)
    target = torch.randn(2, 3, 6)
    vision_features = torch.randn(2, 256, 2048)
    text_features = torch.randn(2, 765, 2048)
    predictions = builder.predict_bboxes(vision_features, text_features)
    masks = torch.ones(2, 3, dtype=torch.bool)
    bbox_loss = builder.compute_bbox_loss(predictions, target, masks)
