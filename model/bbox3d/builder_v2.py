import torch 
from torch import nn
from typing import Optional
import torch.nn.functional as F
from einops import rearrange
from types import SimpleNamespace


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
            from model.bbox3d.builder import build_bbox3d_module

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
            # Fallback loss calculator
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

            # Combine features
            combined = torch.cat([vision_pooled, text_pooled], dim=-1)

            # Predict bboxes
            return self.bbox3d_head(combined)
        except Exception as e:
            print(f"Warning: Failed to predict bboxes: {e}")
            return None

    def compute_bbox_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute bbox prediction loss with proper shape handling"""
        if not self.enabled or predictions is None:
            return torch.tensor(
                0.0, device=predictions.device if predictions is not None else "cpu"
            )

        try:
            # Handle different prediction shapes
            if predictions.dim() == 2 and targets.dim() == 3:
                # Pred: [B, 6], Target: [B, max_boxes, 6]
                if masks is not None:
                    # Use first valid box as target
                    valid_indices = masks.argmax(dim=1)  # Get first valid box index
                    targets = targets[torch.arange(targets.size(0)), valid_indices]
                else:
                    targets = targets[:, 0, :]  # Use first box

            elif predictions.dim() == 3 and targets.dim() == 3:
                # Both: [B, max_boxes, 6]
                if masks is not None:
                    loss_full = F.smooth_l1_loss(predictions, targets, reduction="none")
                    loss_masked = loss_full * masks.unsqueeze(-1).float()
                    return loss_masked.sum() / masks.sum().clamp(min=1)

            return self.loss_calculator.compute_loss(predictions, targets, masks)
        except Exception as e:
            print(f"Warning: Failed to compute bbox loss: {e}")
            return torch.tensor(0.0, device=predictions.device)