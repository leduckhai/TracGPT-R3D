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
from model.bbox3d.helper import box3d_iou_single

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

    def predict_bboxes(
        self, vision_features: torch.Tensor, text_features: torch.Tensor=None
    ) -> torch.Tensor:
        """Predict 3D bounding boxes"""
        if not self.enabled:
            return None

        try:
            vision_pooled = vision_features.mean(dim=1)  # [B, D_v]
            # text_pooled = text_features.mean(dim=1)  # [B, D_t]

            # combined = torch.cat([vision_pooled, text_pooled], dim=-1)
            # return self.bbox3d_head(combined)
            return self.bbox3d_head(vision_pooled)
        except Exception as e:
            raise Exception(f"Warning: Failed to predict bboxes: {e}")

    def compute_bbox_loss(
        self,
        bbox_preds: torch.Tensor,
        targets: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optimized bbox loss computation with vectorized operations"""
        # total_loss = 0.0
        total_loss=[]
        valid_batches = 0
        
        ious=[]
        for b in range(len(bbox_preds)):
            bbox_pred = bbox_preds[b]  # [num_preds, 6]
            mask = masks[b]  # [max_num_gt]
            target = targets[b]  # [max_num_gt, 6]
            gt_boxes = target[mask]  # [num_valid_gt, 6]
            if len(gt_boxes) == 0:
                continue
            gt_boxes = gt_boxes[:1]  # [1, 6]
            mask = mask[:1]  # [1]
            # Convert format if needed
            print("before convert_model_to_gt_format bbox_pred", bbox_pred.tolist())
            pred_boxes_minmax = convert_model_to_gt_format(bbox_pred)
            print("pred_boxes_minmax ", pred_boxes_minmax.tolist())
            print("gt_boxes ", gt_boxes.tolist())
            # batch_loss = F.smooth_l1_loss(
            #     pred_boxes_minmax,
            #     gt_boxes,
            #     reduction='sum'  # Preserve magnitude
            # )
            l2_reg = 0.01 * torch.sum(pred_boxes_minmax ** 2)
    
            loss = F.mse_loss(pred_boxes_minmax, gt_boxes) + l2_reg
            iou = box3d_iou_single(pred_boxes_minmax.detach().cpu(), gt_boxes.detach().cpu(), denormalize=True)
            print("bbox iou", iou)
            total_loss.append(loss)
        return torch.stack(total_loss).mean() if total_loss else torch.tensor(0.0), 0.0
        #     matches = hungarian_iou_matching(pred_boxes_minmax, gt_boxes)
        #     if matches:
        #         pred_indices = [m[0] for m in matches]
        #         gt_indices = [m[1] for m in matches]
                
        #         for match_pred,match_gt in zip(  pred_boxes_minmax[pred_indices], gt_boxes[gt_indices]):
                    
        #             print("bbox matches",  match_pred, match_gt)
        #             iou = box3d_iou_single(match_pred.detach().cpu(), match_gt.detach().cpu(), denormalize=True)
        #             print("bbox iou", iou)
        #             ious.append(iou.item())
        #         batch_loss = F.smooth_l1_loss(
        #             pred_boxes_minmax[pred_indices],
        #             gt_boxes[gt_indices],
        #             reduction='sum'  # Preserve magnitude
        #         )
        #         total_loss += batch_loss
        #         valid_batches += 1

        # # Normalize by number of valid batches and matches
        return total_loss / max(1, valid_batches), np.mean(ious) if ious else 0.0

    
if __name__ == "__main__":
    import yaml
    from utils.type import dict_to_namespace

    config_path = "config/llama.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    # img feature shape: image features shape torch.Size([2, 256, 2048])
    #  assume bbox in format [center_x, center_y, center_z, width, height, length]
    builder = BBox3DPredictor(config.tiny_llama.bbox_predictor)
    # target = torch.randn(2, 3, 6)
    target=torch.tensor([0.2125, 0.4763, 0.0010, 0.4767, 0.7216, 0.8750]).unsqueeze(0)
    vision_features = torch.randn(1, 256, 2048)
    text_features = torch.randn(1, 765, 2048)
    predictions = builder.predict_bboxes(vision_features, text_features)
    print("predictions shape", predictions.shape,target.shape)
    masks = torch.ones(1, 6, dtype=torch.bool)
    bbox_loss = builder.compute_bbox_loss(predictions, target, masks)
