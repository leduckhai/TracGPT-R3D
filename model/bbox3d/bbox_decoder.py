import os
import sys
from dotenv import load_dotenv
import torch
from torch import nn

class BBox3DDecoder:
    def __init__(self, conf_threshold=0.5, nms_threshold=0.5):
        """
        3D Bounding Box Decoder for patch-based predictions
        
        Args:
            conf_threshold: Confidence threshold for filtering predictions
            nms_threshold: IoU threshold for Non-Maximum Suppression
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
    
    def decode_predictions(self, preds, class_preds=None, original_shape=None):
        """
        Decode model predictions into meaningful 3D bounding boxes
        
        Args:
            preds: Tuple of (delta_zxy, log_dwh, conf)
                - delta_zxy: (B, num_anchors, 3, D, H, W) - position offsets
                - log_dwh: (B, num_anchors, 3, D, H, W) - log-scale dimensions
                - conf: (B, num_anchors, D, H, W) - confidence scores
            class_preds: (B, num_anchors, num_classes, D, H, W) - class predictions (optional)
            original_shape: (depth, height, width) - original volume shape for scaling
            
        Returns:
            List of decoded boxes for each batch item:
            Each item contains:
                - boxes: (N, 6) tensor [cx, cy, cz, w, h, d] in normalized coords
                - scores: (N,) tensor of confidence scores
                - classes: (N,) tensor of predicted class indices (if class_preds provided)
        """
        delta_zxy, log_dwh, conf_pred = preds
        B, num_anchors, _, D, H, W = delta_zxy.shape
        device = delta_zxy.device
        
        # Create grid coordinates for each patch center
        z_centers = torch.linspace(0, 1, D, device=device).view(1, 1, D, 1, 1)
        y_centers = torch.linspace(0, 1, H, device=device).view(1, 1, 1, H, 1)
        x_centers = torch.linspace(0, 1, W, device=device).view(1, 1, 1, 1, W)
        
        # Decode position: grid_center + offset
        pred_cx = x_centers + delta_zxy[:, :, 1, ...]  # x offset
        pred_cy = y_centers + delta_zxy[:, :, 0, ...]  # y offset  
        pred_cz = z_centers + delta_zxy[:, :, 2, ...]  # z offset
        
        # Decode dimensions: exp(log_scale)
        pred_w = torch.exp(log_dwh[:, :, 1, ...])  # width
        pred_h = torch.exp(log_dwh[:, :, 0, ...])  # height
        pred_d = torch.exp(log_dwh[:, :, 2, ...])  # depth
        
        # Stack into box format: [cx, cy, cz, w, h, d]
        pred_boxes = torch.stack([pred_cx, pred_cy, pred_cz, pred_w, pred_h, pred_d], dim=2)
        # Shape: (B, num_anchors, 6, D, H, W)
        
        # Process class predictions if provided
        if class_preds is not None:
            class_scores, class_indices = torch.max(class_preds, dim=2)
            # Shape: (B, num_anchors, D, H, W)
        
        batch_results = []
        
        for b in range(B):
            # Get predictions for this batch item
            boxes_b = pred_boxes[b]  # (num_anchors, 6, D, H, W)
            conf_b = conf_pred[b]    # (num_anchors, D, H, W)
            
            # Reshape to flatten spatial dimensions
            boxes_flat = boxes_b.permute(0, 2, 3, 4, 1).reshape(-1, 6)  # (num_anchors*D*H*W, 6)
            conf_flat = conf_b.flatten()  # (num_anchors*D*H*W,)
            
            # Filter by confidence threshold
            valid_mask = conf_flat > self.conf_threshold
            if not valid_mask.any():
                # No valid predictions
                batch_results.append({
                    'boxes': torch.empty(0, 6, device=device),
                    'scores': torch.empty(0, device=device),
                    'classes': torch.empty(0, dtype=torch.long, device=device) if class_preds is not None else None
                })
                continue
            
            valid_boxes = boxes_flat[valid_mask]
            valid_scores = conf_flat[valid_mask]
            
            # Handle class predictions if provided
            if class_preds is not None:
                class_scores_b = class_scores[b].flatten()
                class_indices_b = class_indices[b].flatten()
                valid_class_scores = class_scores_b[valid_mask]
                valid_class_indices = class_indices_b[valid_mask]
                # Combine confidence and class scores
                final_scores = valid_scores * valid_class_scores
            else:
                final_scores = valid_scores
                valid_class_indices = None
            
            # Apply Non-Maximum Suppression
            if len(valid_boxes) > 0:
                keep_indices = self.nms_3d(valid_boxes, final_scores, self.nms_threshold)
                final_boxes = valid_boxes[keep_indices]
                final_scores = final_scores[keep_indices]
                final_classes = valid_class_indices[keep_indices] if valid_class_indices is not None else None
            else:
                final_boxes = torch.empty(0, 6, device=device)
                final_scores = torch.empty(0, device=device)
                final_classes = torch.empty(0, dtype=torch.long, device=device) if class_preds is not None else None
            
            # Scale to original volume size if provided
            if original_shape is not None:
                final_boxes = self.scale_to_original(final_boxes, original_shape)
            
            batch_results.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'classes': final_classes
            })
        
        return batch_results
    
    def nms_3d(self, boxes, scores, threshold):
        """
        3D Non-Maximum Suppression
        
        Args:
            boxes: (N, 6) [cx, cy, cz, w, h, d]
            scores: (N,) confidence scores
            threshold: IoU threshold
            
        Returns:
            keep_indices: indices of boxes to keep
        """
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Sort by scores (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep = []
        while len(sorted_indices) > 0:
            # Keep the highest scoring box
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx:current_idx+1]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self.compute_iou_3d(current_box, remaining_boxes)
            
            # Remove boxes with IoU > threshold
            mask = ious.squeeze() <= threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)
    
    def compute_iou_3d(self, boxes1, boxes2):
        """
        Compute 3D IoU between two sets of boxes
        
        Args:
            boxes1: (N1, 6) [cx, cy, cz, w, h, d]
            boxes2: (N2, 6) [cx, cy, cz, w, h, d]
            
        Returns:
            iou: (N1, N2) IoU matrix
        """
        # Convert center format to corner format
        boxes1_corners = self.center_to_corners(boxes1)
        boxes2_corners = self.center_to_corners(boxes2)
        
        # boxes1_corners: (N1, 6) [x1, y1, z1, x2, y2, z2]
        # boxes2_corners: (N2, 6) [x1, y1, z1, x2, y2, z2]
        
        # Calculate intersection
        x1 = torch.max(boxes1_corners[:, 0:1], boxes2_corners[:, 0:1].T)
        y1 = torch.max(boxes1_corners[:, 1:2], boxes2_corners[:, 1:2].T)
        z1 = torch.max(boxes1_corners[:, 2:3], boxes2_corners[:, 2:3].T)
        x2 = torch.min(boxes1_corners[:, 3:4], boxes2_corners[:, 3:4].T)
        y2 = torch.min(boxes1_corners[:, 4:5], boxes2_corners[:, 4:5].T)
        z2 = torch.min(boxes1_corners[:, 5:6], boxes2_corners[:, 5:6].T)
        
        # Intersection volume
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0) * torch.clamp(z2 - z1, min=0)
        
        # Calculate volumes
        vol1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]  # w * h * d
        vol2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]  # w * h * d
        
        # Union volume
        union = vol1.unsqueeze(1) + vol2.unsqueeze(0) - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        return iou
    
    def center_to_corners(self, boxes):
        """
        Convert center format to corner format
        
        Args:
            boxes: (N, 6) [cx, cy, cz, w, h, d]
            
        Returns:
            corners: (N, 6) [x1, y1, z1, x2, y2, z2]
        """
        cx, cy, cz, w, h, d = boxes.unbind(dim=1)
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        z1 = cz - d / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        z2 = cz + d / 2
        
        return torch.stack([x1, y1, z1, x2, y2, z2], dim=1)
    
    def scale_to_original(self, boxes, original_shape):
        """
        Scale normalized boxes to original volume dimensions
        
        Args:
            boxes: (N, 6) [cx, cy, cz, w, h, d] in normalized coords [0,1]
            original_shape: (depth, height, width) original volume shape
            
        Returns:
            scaled_boxes: (N, 6) boxes in original coordinate system
        """
        if len(boxes) == 0:
            return boxes
        
        depth, height, width = original_shape
        scaling_factors = torch.tensor([width, height, depth, width, height, depth], 
                                     device=boxes.device, dtype=boxes.dtype)
        
        return boxes * scaling_factors.unsqueeze(0)

def example_usage():
    """Example of how to use the decoder"""
    
    # Create decoder
    decoder = BBox3DDecoder(conf_threshold=0.5, nms_threshold=0.4)
    
    # Example predictions (from model)
    B, num_anchors, D, H, W = 2, 3, 16, 32, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock predictions
    delta_zxy = torch.randn(B, num_anchors, 3, D, H, W, device=device)
    log_dwh = torch.randn(B, num_anchors, 3, D, H, W, device=device)
    conf = torch.sigmoid(torch.randn(B, num_anchors, D, H, W, device=device))
    class_preds = torch.softmax(torch.randn(B, num_anchors, 25, D, H, W, device=device), dim=2)
    
    preds = (delta_zxy, log_dwh, conf)
    
    # Decode predictions
    results = decoder.decode_predictions(preds, class_preds, original_shape=(64, 128, 128))
    
    # Process results
    for i, result in enumerate(results):
        print(f"Batch {i}:")
        print(f"  Detected {len(result['boxes'])} objects")
        print(f"  Boxes shape: {result['boxes'].shape}")
        print(f"  Scores shape: {result['scores'].shape}")
        if result['classes'] is not None:
            print(f"  Classes shape: {result['classes'].shape}")
        print()

if __name__ == "__main__":
    example_usage()