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
from model.bbox3d.helper import box3d_iou_single,iou_loss,diou_3d

from model.bbox3d.bbox_head_white import AnchorBBox3DLossV2,AnchorBBox3DHeadV2
        

        
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
      
        if self.config.bbox_type=="anchor_v2":
            self.bbox3d_head=AnchorBBox3DHeadV2(self.config)
            self.loss_calculator=AnchorBBox3DLossV2()
        else:
            raise NotImplementedError
      
        # self.loss_calculator = self._create_loss_calculator()
        self.enabled = True

    def predict_bboxes(
        self, vision_features: torch.Tensor, text_features: torch.Tensor=None
    ) -> torch.Tensor:
        """Predict 3D bounding boxes"""
        if not self.enabled:
            return None
        if self.config.bbox_type == "anchor_v2":
            return self.bbox3d_head(vision_features)
        else:

            raise Exception(f"Warning: Failed to predict bboxes: {e}")

    def compute_bbox_loss(
        self,
        # bbox_preds,
        delta_xyz: torch.Tensor,
        log_dwh: torch.Tensor,
        conf_pred: torch.Tensor,
        targets: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Optimized bbox loss computation with vectorized operations"""
        if self.config.bbox_type=="anchor" or self.config.bbox_type=="anchor_v2":
            
            return self.loss_calculator(delta_xyz, log_dwh, conf_pred, targets,masks)
      
        else:
            raise NotImplementedError(
                f"Loss calculation for bbox type {self.config.bbox_type} is not implemented."   )
    
       
    
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
