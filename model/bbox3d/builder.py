
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class BBox3DHead(nn.Module):
    """3D Bounding Box prediction head"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # 3D bbox regression: center (x,y,z) + dimensions (w,h,l) + rotation (yaw)
        self.bbox_head = nn.Linear(hidden_dim, 7)  # 3 + 3 + 1
        
        # Classification head (if needed)
        self.cls_head = nn.Linear(hidden_dim, num_classes) if num_classes > 1 else None
        
        # Confidence/objectness head
        self.conf_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input features [batch_size, input_dim]
        Returns:
            Dict containing bbox predictions, confidence, and optionally classification
        """
        features = self.feature_extractor(x)
        
        # 3D bbox prediction: [cx, cy, cz, w, h, l, yaw]
        bbox_pred = self.bbox_head(features)
        
        # Confidence score
        conf_pred = torch.sigmoid(self.conf_head(features))
        
        outputs = {
            'bbox_pred': bbox_pred,
            'conf_pred': conf_pred,
        }
        
        # Classification if multi-class
        if self.cls_head is not None:
            cls_pred = self.cls_head(features)
            outputs['cls_pred'] = cls_pred
        
        return outputs


class PointNetBBox3D(nn.Module):
    """PointNet-based 3D bbox module for point cloud input"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 512, num_classes: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Point-wise feature extraction
        self.point_conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # 3D bbox head
        self.bbox_head = BBox3DHead(hidden_dim, hidden_dim, num_classes)
    
    def forward(self, points):
        """
        Forward pass for point cloud
        Args:
            points: [batch_size, num_points, input_dim] or [batch_size, input_dim, num_points]
        """
        if points.dim() == 3 and points.size(-1) == self.input_dim:
            points = points.transpose(1, 2)  # [B, C, N]
        
        # Point-wise features
        point_features = self.point_conv(points)  # [B, hidden_dim, N]
        
        # Global feature
        global_features = self.global_pool(point_features).squeeze(-1)  # [B, hidden_dim]
        
        # 3D bbox prediction
        return self.bbox_head(global_features)


class VisionBBox3D(nn.Module):
    """Vision-based 3D bbox module for image features"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 1):
        super().__init__()
        self.input_dim = input_dim
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # 3D bbox head
        self.bbox_head = BBox3DHead(hidden_dim, hidden_dim, num_classes)
    
    def forward(self, vision_features):
        """
        Forward pass for vision features
        Args:
            vision_features: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        """
        if vision_features.dim() == 3:
            # Pool sequence dimension if present
            vision_features = vision_features.mean(dim=1)
        
        processed_features = self.feature_processor(vision_features)
        return self.bbox_head(processed_features)


def build_bbox3d_module(config) -> nn.Module:
    """
    Builder function for 3D bounding box modules
    
    Args:
        config: Configuration object with bbox3d_module settings
        
    Returns:
        nn.Module: The built 3D bbox module
    """
    bbox3d_config = getattr(config, 'bbox3d_module', {})
    
    # Get module type
    module_type = bbox3d_config.get('type', 'vision')  # 'vision', 'pointnet', or 'simple'
    
    # Get common parameters
    input_dim = bbox3d_config.get('input_dim', getattr(config, 'hidden_size', 768))
    hidden_dim = bbox3d_config.get('hidden_dim', 512)
    num_classes = bbox3d_config.get('num_classes', 1)
    
    if module_type == 'pointnet':
        # For point cloud inputs
        point_dim = bbox3d_config.get('point_dim', 3)  # x, y, z coordinates
        return PointNetBBox3D(
            input_dim=point_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
    
    elif module_type == 'vision':
        # For vision feature inputs
        return VisionBBox3D(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
    
    elif module_type == 'simple':
        # Simple 3D bbox head
        return BBox3DHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"Unknown bbox3d module type: {module_type}")


# Example configuration usage:
"""
config.bbox3d_module = {
    'type': 'vision',  # or 'pointnet', 'simple'
    'input_dim': 768,  # will use config.hidden_size if not specified
    'hidden_dim': 512,
    'num_classes': 1,  # for single-class detection
    'point_dim': 3,  # only for pointnet type
}
"""