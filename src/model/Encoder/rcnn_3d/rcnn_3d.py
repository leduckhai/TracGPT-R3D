import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet
import sys 
sys.path.append("/root/TracGPT-R3D")
from src.model.Encoder.rcnn_3d.sample_prosal import _sample_proposals
# ------------------------------
# 3D ResNet backbone (feature extractor)
# ------------------------------
class ResNet3DFeature(nn.Module):
    def __init__(self, block='basic', layers=[2,2,2,2], block_inplanes=[64,128,256,512], n_input_channels=1):
        super().__init__()
        self.backbone = ResNet(
            block=block,
            layers=layers,
            block_inplanes=block_inplanes,
            spatial_dims=3,
            n_input_channels=n_input_channels
        )
        
    def forward(self, x):
        # Forward through all layers except avgpool and fc
        for name, layer in self.backbone.named_children():
            if name in ["avgpool", "fc"]:
                break
            x = layer(x)
        return x  # (B, 512, D', H', W')

# ------------------------------
# 3D Anchor Generator
# ------------------------------
class AnchorGenerator3D:
    def __init__(self, sizes=((8,8,8),(16,16,16)), aspect_ratios=((1,1,1),)):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
    
    def __call__(self, feature_map):
        device = feature_map.device
        anchors = []
        D, H, W = feature_map.shape[2:]
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    for size in self.sizes:
                        for ar in self.aspect_ratios:
                            d, h, w = size
                            anchor = [z, y, x, d, h, w]  # center + size
                            anchors.append(torch.tensor(anchor, device=device, dtype=torch.float32))
        anchors = torch.stack(anchors, dim=0)
        return anchors  # shape: (num_anchors, 6)

class RPN3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.anchor_generator = AnchorGenerator3D()
        self.conv = nn.Conv3d(in_channels, 512, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv3d(512, 3*2, kernel_size=1)  # 3 anchors
        self.bbox_pred = nn.Conv3d(512, 3*6, kernel_size=1)   # 6 coords
    
    def forward(self, x, targets=None):
        print("RPN input shape:", x.shape)
        anchors = self.anchor_generator(x)
        print("Generated anchors shape:", anchors.shape)
        features = F.relu(self.conv(x))
        
        # Dummy proposals for testing
        N = 64
        B = x.shape[0]
        rois = torch.zeros((N, 6), device=x.device)  # [z1,y1,x1,z2,y2,x2]
        roi_labels = torch.randint(0, 2, (N,), device=x.device)
        rpn_loss = torch.tensor(0.0)
        return rois, roi_labels, rpn_loss

# ------------------------------
# 3D RoI Align (stub for testing)
# ------------------------------
class RoIAlign3D(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, features, rois):
        # features: (B, C, D, H, W), rois: (N, 7) or (N,6)
        N, C = rois.shape[0], features.shape[1]
        pooled = torch.randn((N, C, self.output_size, self.output_size, self.output_size), device=features.device)
        return pooled

# ------------------------------
# Detection Head
# ------------------------------
class DetectionHead3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes*6)
    
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x)

# ------------------------------
# Full FasterRCNN3D
# ------------------------------
class FasterRCNN3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = ResNet3DFeature()
        self.rpn = RPN3D(512)
        self.roi_pool = RoIAlign3D(output_size=7)
        self.head = DetectionHead3D(512, num_classes)
    
    def forward(self, x, targets=None):
        features = self.backbone(x)
        rois, roi_labels, rpn_loss = self.rpn(features, targets)
        
        # Add batch_idx to ROIs for RoIAlign
        batch_idx = torch.zeros(rois.shape[0], 1, device=x.device)
        rois_with_batch = torch.cat([batch_idx, rois], dim=1)  # (N,7)
        
        pooled_features = self.roi_pool(features, rois_with_batch)
        cls_logits, box_preds = self.head(pooled_features)
        return {
            "rpn_loss": rpn_loss,
            "cls_logits_shape": cls_logits.shape,
            "box_preds_shape": box_preds.shape
        }

# ------------------------------
# Test run
# ------------------------------
if __name__ == "__main__":
    x = torch.randn(2,1,32,256,256)
    model = FasterRCNN3D()
    out = model(x)
    print(out)

# class ResNet3DFeature(nn.Module):
#     def __init__(self, block='basic', layers=[2,2,2,2], block_inplanes=[64,128,256,512], n_input_channels=1):
#         super().__init__()
#         self.backbone = ResNet(
#             block=block,
#             layers=layers,
#             block_inplanes=block_inplanes,
#             spatial_dims=3,
#             n_input_channels=n_input_channels
#         )
    
#     def forward(self, x):
#         # Manually forward through layers to skip final pooling/fc
#         for name, layer in self.backbone.named_children():
#             if name in ["avgpool", "fc"]:
#                 break  # stop before global pooling and FC
#             x = layer(x)
#         return x  # shape: (B, 512, D', H', W')
    
# class FasterRCNN3D(nn.Module):
#     def __init__(self, num_classes=2, backbone='resnet18_3d'):
#         super().__init__()
        
#         # Backbone (3D version)
#         if backbone == 'resnet18_3d':
#             self.backbone = ResNet3DFeature()
#             in_channels = 512
#         else:
#             raise ValueError(f"Unsupported backbone: {backbone}")
        
#         # Region Proposal Network (3D)
#         self.rpn = RPN3D(in_channels)
        
#         # ROI Pooling (3D)
#         self.roi_pool = RoIAlign3D(output_size=7, spatial_scale=1.0)
        
#         # Detection Head
#         self.head = DetectionHead3D(in_channels, num_classes)
        
#     def forward(self, x, targets=None):
#         # x: (B, C, D, H, W)
#         features = self.backbone(x)  # (B, 512, D/16, H/16, W/16)
#         print("Features shape:", features.shape)
#         # RPN proposals
#         proposals, rpn_loss = self.rpn(features, targets)
#         print("Proposals shape:", proposals.shape)
#         if self.training:
#             # Sample ROIs for training
#             rois, roi_labels = self._sample_proposals(proposals, targets)
#             print("Sampled ROIs shape:", rois.shape, "Labels shape:", roi_labels.shape)
#             pooled_features = self.roi_pool(features, rois)
#             cls_logits, box_preds = self.head(pooled_features)
            
#             # Compute losses
#             cls_loss = F.cross_entropy(cls_logits, roi_labels)
#             box_loss = F.smooth_l1_loss(box_preds, targets['boxes'])
            
#             return {'rpn_loss': rpn_loss, 
#                     'cls_loss': cls_loss, 
#                     'box_loss': box_loss}
#         else:
#             # Inference
#             pooled_features = self.roi_pool(features, proposals)
#             cls_logits, box_preds = self.head(pooled_features)
#             return self._postprocess(proposals, cls_logits, box_preds)
        
#     def _sample_proposals(self, proposals, targets):
#         # Simple placeholder: take first N proposals
#         N = min(len(proposals), 64)
#         rois = proposals[:N]
#         roi_labels = torch.randint(0, 2, (N,), device=rois.device)
#         return rois, roi_labels
    
#     def _postprocess(self, proposals, cls_logits, box_preds):
#         # Decode boxes (placeholder: just return)
#         return {'boxes': box_preds, 'scores': F.softmax(cls_logits, dim=-1)}

# # --------------------------
# # RPN 3D
# # --------------------------
# class RPN3D(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.anchor_generator = AnchorGenerator3D()
#         self.conv = nn.Conv3d(in_channels, 512, kernel_size=3, padding=1)
#         self.cls_logits = nn.Conv3d(512, 3*2, kernel_size=1)  # 3 anchors
#         self.bbox_pred = nn.Conv3d(512, 3*6, kernel_size=1)   # 6 coords
        
#     def forward(self, x, targets):
#         print("RPN input shape:", x.shape)
#         # RPN input shape: torch.Size([2, 512, 2, 16, 16])
#         anchors = self.anchor_generator(x)
#         print("Generated anchors shape:", anchors.shape)
#         features = F.relu(self.conv(x))
#         logits = self.cls_logits(features)
#         bbox_reg = self.bbox_pred(features)
        
#         if self.training:
#             # placeholder loss
#             rpn_loss = torch.tensor(0.0, device=x.device)
#             return anchors, rpn_loss
#         else:
#             return self._decode_boxes(anchors, bbox_reg)
    
#     def _decode_boxes(self, anchors, bbox_reg):
#         # simple: add deltas to anchors
#         return anchors + bbox_reg.flatten(1,2)


# class RoIAlign3D(nn.Module):
#     def __init__(self, output_size, spatial_scale):
#         super().__init__()
#         self.output_size = output_size
#         self.spatial_scale = spatial_scale
        
#     def forward(self, features, rois):
#         B, C, D, H, W = features.shape
#         pooled = []
#         for roi in rois:
#             b_idx = int(roi[0])
#             z1, y1, x1, z2, y2, x2 = roi[1:].long()
#             z1 = max(z1, 0); y1 = max(y1, 0); x1 = max(x1, 0)
#             z2 = min(z2, D-1); y2 = min(y2, H-1); x2 = min(x2, W-1)
#             roi_feat = features[b_idx, :, z1:z2+1, y1:y2+1, x1:x2+1]
#             pooled_feat = F.adaptive_max_pool3d(roi_feat, self.output_size)
#             pooled.append(pooled_feat)
#         return torch.stack(pooled)

# # --------------------------
# # Anchor Generator 3D
# # --------------------------
# class AnchorGenerator3D:
#     def __init__(self, sizes=((32,32,32),(64,64,64)), aspect_ratios=((1,1,1),)):
#         self.sizes = sizes
#         self.aspect_ratios = aspect_ratios

#     def __call__(self, feature_map):
#         device = feature_map.device
#         anchors = []

#         for z in range(feature_map.shape[2]):
#             for y in range(feature_map.shape[3]):
#                 for x in range(feature_map.shape[4]):
#                     for size in self.sizes:
#                         for ar in self.aspect_ratios:
#                             anchor = self._make_anchor(z, y, x, size, ar)
#                             anchors.append(torch.tensor(anchor, device=device, dtype=torch.float32))

#         # Stack all anchors into a single tensor
#         anchors = torch.stack(anchors, dim=0)  # shape: (num_anchors, 6)
#         return anchors

#     def _make_anchor(self, z, y, x, size, ar):
#         # size and ar are tuples of (d,h,w)
#         d, h, w = size
#         anchor = [z, y, x, d, h, w]  # centers + size
#         return anchor
# # --------------------------
# # Detection Head 3D
# # --------------------------
# class DetectionHead3D(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         self.fc1 = nn.Linear(in_channels * 7*7*7, 1024)
#         self.fc2 = nn.Linear(1024, 1024)
#         self.cls_score = nn.Linear(1024, num_classes)
#         self.bbox_pred = nn.Linear(1024, num_classes*6)
        
#     def forward(self, x):
#         x = x.flatten(1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.cls_score(x), self.bbox_pred(x)

# if __name__ == "__main__":
#     model = FasterRCNN3D(num_classes=2)
#     x = torch.randn(2, 1, 32, 256, 256)  # Batch of 2, single channel, 32 slices
#     output = model(x)
#     print(output.keys())  # Should contain 'boxes' and 'scores'