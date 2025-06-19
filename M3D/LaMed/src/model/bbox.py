import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import math


class VisionFeatureExtractor(nn.Module):
    """
    Extracts vision features for bbox prediction.
    Supports different backbone architectures and feature extraction strategies.
    """
    
    def __init__(self, 
                 backbone_type='resnet50', 
                 feature_dim=2048,
                 spatial_pool='adaptive',
                 pretrained=True,
                 freeze_backbone=False):
        super().__init__()
        
        self.backbone_type = backbone_type
        self.feature_dim = feature_dim
        self.spatial_pool = spatial_pool
        
        # Initialize backbone
        if backbone_type == 'resnet50':
            if pretrained:
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.backbone = resnet50(weights=None)
            
            # Remove final classification layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_output_dim = 2048
            
        elif backbone_type == 'clip_vit':
            # Example for CLIP ViT (you'd need to install clip-by-openai)
            # import clip
            # self.backbone, _ = clip.load("ViT-B/32", device="cpu")
            # backbone_output_dim = 512
            raise NotImplementedError("CLIP ViT backbone not implemented in this example")
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Spatial pooling layer
        if spatial_pool == 'adaptive':
            self.spatial_pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif spatial_pool == 'max':
            self.spatial_pooling = nn.AdaptiveMaxPool2d((1, 1))
        elif spatial_pool == 'attention':
            self.spatial_pooling = SpatialAttentionPooling(backbone_output_dim)
        else:
            self.spatial_pooling = None
        
        # Feature projection to desired dimension
        if backbone_output_dim != feature_dim:
            self.feature_projection = nn.Linear(backbone_output_dim, feature_dim)
        else:
            self.feature_projection = nn.Identity()
    
    def forward(self, images):
        """
        Extract vision features from images
        
        Args:
            images: [B, C, H, W] tensor of images
            
        Returns:
            vision_features: [B, feature_dim] tensor of vision features
        """
        batch_size = images.size(0)
        
        # Extract features using backbone
        features = self.backbone(images)  # [B, C, H, W]
        
        # Apply spatial pooling
        if self.spatial_pooling is not None:
            if isinstance(self.spatial_pooling, SpatialAttentionPooling):
                pooled_features = self.spatial_pooling(features)  # [B, C]
            else:
                pooled_features = self.spatial_pooling(features)  # [B, C, 1, 1]
                pooled_features = pooled_features.view(batch_size, -1)  # [B, C]
        else:
            # Global average pooling as fallback
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
            pooled_features = pooled_features.view(batch_size, -1)
        
        # Project to desired feature dimension
        vision_features = self.feature_projection(pooled_features)
        
        return vision_features


class SpatialAttentionPooling(nn.Module):
    """Attention-based spatial pooling for better feature aggregation"""
    
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 4
        
        self.attention = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        attention_weights = self.attention(x)  # [B, 1, H, W]
        weighted_features = x * attention_weights  # [B, C, H, W]
        pooled = weighted_features.sum(dim=[2, 3])  # [B, C]
        return pooled


class MultiScaleVisionFeatureExtractor(nn.Module):
    """
    Multi-scale vision feature extractor for better spatial understanding
    """
    
    def __init__(self, backbone_type='resnet50', feature_dim=2048):
        super().__init__()
        
        if backbone_type == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Extract different layers for multi-scale features
            self.layer1 = nn.Sequential(*list(backbone.children())[:5])  # Early features
            self.layer2 = nn.Sequential(*list(backbone.children())[5:6])  # Mid features  
            self.layer3 = nn.Sequential(*list(backbone.children())[6:7])  # Late features
            self.layer4 = nn.Sequential(*list(backbone.children())[7:8])  # Final features
            
            # Feature dimensions for each layer
            self.layer_dims = [256, 512, 1024, 2048]
        
        # Fusion network to combine multi-scale features
        total_dim = sum(self.layer_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, images):
        features = []
        
        # Extract multi-scale features
        x = self.layer1(images)
        features.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
        
        x = self.layer2(x)
        features.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
        
        x = self.layer3(x)
        features.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
        
        x = self.layer4(x)
        features.append(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))
        
        # Concatenate and fuse features
        combined_features = torch.cat(features, dim=1)
        fused_features = self.fusion(combined_features)
        
        return fused_features


class BboxHead(nn.Module):
    """
    Bbox prediction head with multiple architecture options
    """
    
    def __init__(self, 
                 input_dim, 
                 bbox_dim=6,  # 6 for 3D bbox: [x_min, x_max, y_min, y_max, z_min, z_max]
                 max_bboxes=1,
                 head_type='simple',
                 dropout=0.1,
                 use_layer_norm=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.bbox_dim = bbox_dim
        self.max_bboxes = max_bboxes
        self.head_type = head_type
        
        if head_type == 'simple':
            self.bbox_head = self._build_simple_head(input_dim, bbox_dim * max_bboxes, dropout)
        elif head_type == 'mlp':
            self.bbox_head = self._build_mlp_head(input_dim, bbox_dim * max_bboxes, dropout)
        elif head_type == 'transformer':
            self.bbox_head = self._build_transformer_head(input_dim, bbox_dim, max_bboxes, dropout)
        elif head_type == 'residual':
            self.bbox_head = self._build_residual_head(input_dim, bbox_dim * max_bboxes, dropout)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
    
    def _build_simple_head(self, input_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout)
        )
    
    def _build_mlp_head(self, input_dim, output_dim, dropout):
        hidden_dim = input_dim // 2
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def _build_transformer_head(self, input_dim, bbox_dim, max_bboxes, dropout):
        """Transformer-based head for handling variable number of bboxes"""
        return TransformerBboxHead(input_dim, bbox_dim, max_bboxes, dropout)
    
    def _build_residual_head(self, input_dim, output_dim, dropout):
        """Residual connections for better gradient flow"""
        return ResidualBboxHead(input_dim, output_dim, dropout)
    
    def forward(self, features):
        """
        Args:
            features: [B, input_dim] combined vision+text features
            
        Returns:
            bbox_pred: [B, max_bboxes, bbox_dim] or [B, bbox_dim] depending on max_bboxes
        """
        if self.use_layer_norm:
            features = self.layer_norm(features)
        
        if self.head_type == 'transformer':
            return self.bbox_head(features)
        else:
            bbox_pred = self.bbox_head(features)  # [B, max_bboxes * bbox_dim]
            
            if self.max_bboxes == 1:
                return bbox_pred  # [B, bbox_dim]
            else:
                batch_size = bbox_pred.size(0)
                return bbox_pred.view(batch_size, self.max_bboxes, self.bbox_dim)


class TransformerBboxHead(nn.Module):
    """Transformer-based bbox head for variable number of bboxes"""
    
    def __init__(self, input_dim, bbox_dim, max_bboxes, dropout=0.1):
        super().__init__()
        
        self.max_bboxes = max_bboxes
        self.bbox_dim = bbox_dim
        
        # Learnable bbox queries
        self.bbox_queries = nn.Parameter(torch.randn(max_bboxes, input_dim))
        
        # Transformer decoder layer
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        
        # Final bbox prediction
        self.bbox_predictor = nn.Linear(input_dim, bbox_dim)
        
        # Initialize bbox queries
        nn.init.xavier_uniform_(self.bbox_queries)
    
    def forward(self, features):
        batch_size = features.size(0)
        
        # Expand bbox queries for batch
        bbox_queries = self.bbox_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Use features as memory (key, value)
        memory = features.unsqueeze(1)  # [B, 1, input_dim]
        
        # Apply transformer
        decoded_features = self.transformer_layer(bbox_queries, memory)
        
        # Predict bboxes
        bbox_pred = self.bbox_predictor(decoded_features)  # [B, max_bboxes, bbox_dim]
        
        return bbox_pred


class ResidualBboxHead(nn.Module):
    """Residual MLP head for better gradient flow"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        
        hidden_dim = input_dim
        
        self.layers = nn.ModuleList([
            ResidualBlock(input_dim, hidden_dim, dropout),
            ResidualBlock(hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, output_dim)
        ])
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


# Example usage and model integration
class VisionTextBboxModel(nn.Module):
    """Complete model integrating vision features and bbox prediction"""
    
    def __init__(self, 
                 text_model_dim=768,
                 vision_feature_dim=2048,
                 bbox_dim=6,
                 max_bboxes=32,
                 vision_backbone='resnet50',
                 bbox_head_type='mlp'):
        super().__init__()
        
        # Vision feature extractor
        self.vision_extractor = VisionFeatureExtractor(
            backbone_type=vision_backbone,
            feature_dim=vision_feature_dim,
            spatial_pool='adaptive'
        )
        
        # Bbox prediction head
        combined_dim = text_model_dim + vision_feature_dim
        self.bbox_3d_head = BboxHead(
            input_dim=combined_dim,
            bbox_dim=bbox_dim,
            max_bboxes=max_bboxes,
            head_type=bbox_head_type,
            dropout=0.1
        )
        
        self.bbox3d_enable = True
    
    def extract_vision_features(self, images):
        """Extract vision features from input images"""
        return self.vision_extractor(images)
    
    def predict_bboxes(self, vision_features, text_features):
        """Predict bboxes from combined features"""
        combined_features = torch.cat([vision_features, text_features], dim=-1)
        return self.bbox_3d_head(combined_features)


# Example initialization
def create_bbox_model(config):
    """Factory function to create bbox model with different configurations"""
    
    model = VisionTextBboxModel(
        text_model_dim=config.get('text_dim', 768),
        vision_feature_dim=config.get('vision_dim', 2048),
        bbox_dim=config.get('bbox_dim', 6),
        max_bboxes=config.get('max_bboxes', 32),
        vision_backbone=config.get('vision_backbone', 'resnet50'),
        bbox_head_type=config.get('bbox_head_type', 'mlp')
    )
    
    return model


# Usage example:
if __name__ == "__main__":
    # Configuration
    config = {
        'text_dim': 768,
        'vision_dim': 2048,
        'bbox_dim': 6,
        'max_bboxes': 32,
        'vision_backbone': 'resnet50',
        'bbox_head_type': 'transformer'
    }
    
    # Create model
    model = create_bbox_model(config)
    
    # Example forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text_features = torch.randn(batch_size, 768)
    
    # Extract vision features
    vision_features = model.extract_vision_features(images)
    print(f"Vision features shape: {vision_features.shape}")
    
    # Predict bboxes
    bbox_pred = model.predict_bboxes(vision_features, text_features)
    print(f"Bbox predictions shape: {bbox_pred.shape}")