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
        self.num_anchors = 1
    
    def decode_predictions_v3(self, center_pred, delta_xyz, log_dwh):
        conf_thresh = self.conf_threshold if hasattr(self, 'conf_threshold') else 0.4
        B = center_pred.shape[0]
        D, H, W = center_pred.shape[1:4]
        device = center_pred.device
        
        # Create grid centers
        z_centers = torch.linspace(0.5/D, 1-0.5/D, D, device=device)
        y_centers = torch.linspace(0.5/H, 1-0.5/H, H, device=device)
        x_centers = torch.linspace(0.5/W, 1-0.5/W, W, device=device)
        
        # Broadcast to all positions
        grid_z, grid_y, grid_x = torch.meshgrid(z_centers, y_centers, x_centers, indexing='ij')
        grid_xyz = torch.stack((grid_x, grid_y, grid_z), dim=-1)  # [D,H,W,3]
        
        batch_results = []
        
        z_dim, y_dim, x_dim = center_pred[0].shape
        z = torch.arange(z_dim)
        y = torch.arange(y_dim)
        x = torch.arange(x_dim)
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')

        combinations = torch.stack((grid_z, grid_y, grid_x), dim=-1).reshape(-1, 3)
        for b in range(B):
        
            valid_boxes = []
            for z, y, x in combinations:
                grid_center = grid_xyz[z,y,x]
                
                for a in range(self.num_anchors):
                    cx = grid_center[0] + delta_xyz[b,z,y,x,a,0]
                    cy = grid_center[1] + delta_xyz[b,z,y,x,a,1]
                    cz = grid_center[2] + delta_xyz[b,z,y,x,a,2]
                    
                    # Decode box dimensions
                    w = torch.exp(log_dwh[b,z,y,x,a,0])
                    h = torch.exp(log_dwh[b,z,y,x,a,1])
                    d = torch.exp(log_dwh[b,z,y,x,a,2])
                    
                    valid_boxes.append(torch.stack([cx, cy, cz, w, h, d]))
            
            batch_results.append({
                'boxes': torch.stack(valid_boxes) if valid_boxes else torch.empty(0, 6, device=device),
                # 'scores': topk_values[:len(valid_boxes)] if valid_boxes else torch.empty(0, device=device)
            })
        
        return batch_results
   
    def nms_3d(self, boxes, scores, threshold):
        print("nms_3d boxes", boxes.shape, "scores", scores.shape)
        """
        3D Non-Maximum Suppression
        
        Args:
            boxes: (N, 6) [cx, cy, cz, w, h, d]
            scores: (N,) confidence scores
            threshold: IoU threshold
            
        Returns:
            keep_indices: indices of boxes to keep
        """
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # Sort boxes by descending scores
        sorted_scores, sorted_indices = scores.sort(descending=True)
        boxes = boxes[sorted_indices]
        
        keep = []
        while len(sorted_indices) > 0:
            # Keep the highest scoring box
            current_idx = 0  # Always the first one since we sorted
            keep.append(sorted_indices[current_idx].item())  # Store the original index
            
            if len(sorted_indices) == 1:
                break
                
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx:current_idx+1]
            remaining_boxes = boxes[current_idx+1:]
            
            ious = self.compute_iou_3d(current_box, remaining_boxes)
            
            # Remove boxes with IoU > threshold
            mask = ious.squeeze() <= threshold
            boxes = boxes[current_idx+1:][mask]
            sorted_indices = sorted_indices[current_idx+1:][mask]
        
        if not keep:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
        # if len(boxes) == 0:
        #     return torch.empty(0, dtype=torch.long, device=boxes.device)
        
        # # Sort by scores (descending)
        # sorted_indices = torch.argsort(scores, descending=True)
        
        # keep = []
        # while len(sorted_indices) > 0:
        #     # Keep the highest scoring box
        #     current_idx = sorted_indices[0]
        #     keep.append(current_idx)
            
        #     if len(sorted_indices) == 1:
        #         break
            
        #     # Calculate IoU with remaining boxes
        #     current_box = boxes[current_idx:current_idx+1]
        #     remaining_boxes = boxes[sorted_indices[1:]]
            
        #     ious = self.compute_iou_3d(current_box, remaining_boxes)
            
        #     # Remove boxes with IoU > threshold
        #     mask = ious.squeeze() <= threshold
        #     sorted_indices = sorted_indices[1:][mask]
        # # print("keep indices", keep)
        # return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)
    
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