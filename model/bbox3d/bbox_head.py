import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


class AnchorBBox3DHead(nn.Module):
    def __init__(self,config,num_anchors=3):
        super().__init__()
        in_channels= config.in_channels if isinstance(config, SimpleNamespace) else in_channels
        self.num_anchors = num_anchors
        self.downsample = nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv3d(
            in_channels, num_anchors * 9, kernel_size=1
        )  # 9 = (Δz,Δx,Δy, log(d),log(w),log(h), conf, cls...)

        # Initialize anchors (example for 3 scales)
        self.register_buffer(
            "anchors",
            torch.tensor(
                [
                    [0.1, 0.1, 0.1],  # Small objects (e.g., 10% of volume size)
                    [0.3, 0.3, 0.3],  # Medium objects
                    [0.5, 0.5, 0.5],  # Large objects
                ]
            ).float(),
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        # B = x.shape[0]
        # # torch.Size([2, 1, 32, 64, 64])
        # out= self.downsample(x)  # Downsample to (B, C, D/2, H/2, W/2)
        # print("Downsampled shape:", out.shape)
        # # [2, 1, 16, 32, 32])
        # out= self.conv(out)  # (B, num_anchors*9, D/2, H/2, W/2)
        # # ([2, 27, 16, 32, 32])
        # print("Conv output shape:", out.shape)
        # out = out.view(
        #     B, self.num_anchors, 9, *out.shape[2:]
        # )  # (B, num_anchors, 9, D/2, H/2, W/2)
        # print("Reshaped output:", out.shape)
        # # Reshaped output: torch.Size([2, 3, 9, 16, 32, 32])
        # # Decode predictions
        # Δzxy = torch.sigmoid(out[..., :3]) * 2 - 0.5  # Δz, Δx, Δy ∈ [-0.5, 1.5]
        # print("Δzxy shape:", Δzxy.shape)
        # # Δzxy shape: torch.Size([2, 3, 9, 16, 32, 3])
        # log_dwh = out[..., 3:6]  # Log-scale dims
        # # Log-scale dimensions shape: torch.Size([2, 3, 9, 16, 32, 3]
        # print("Log-scale dimensions shape:", log_dwh.shape)
        # conf = torch.sigmoid(out[..., 6])
        # print("Confidence shape:", conf.shape)
        # # Confidence shape: torch.Size([2, 3, 9, 16, 32])
        # cls = torch.softmax(out[..., 7:], dim=-1)
        # # 
        # print("Class probabilities shape:", cls.shape)
        # return Δzxy, log_dwh, conf, cls
        B = x.shape[0]  # Batch size
        # Input shape: [B, C, D, H, W] = [2, 1, 32, 64, 64]
        
        # Step 1: Downsample
        out = self.downsample(x)  # [2, 1, 16, 32, 32] (stride=2)
        
        # Step 2: 1x1x1 Conv to predict anchor values
        out = self.conv(out)  # [2, 27, 16, 32, 32] (num_anchors*9=27 channels)
        
        # Step 3: Reshape to separate anchors and predictions
        out = out.view(B, self.num_anchors, 9, *out.shape[2:])  # [2, 3, 9, 16, 32, 32]
        
        # Step 4: Decode predictions (EXPLICIT SLICING)
        Δzxy = torch.sigmoid(out[:, :, :3, :, :, :]) * 2 - 0.5  # [2, 3, 3, 16, 32, 32]
        log_dwh = out[:, :, 3:6, :, :, :]  # [2, 3, 3, 16, 32, 32]
        conf = torch.sigmoid(out[:, :, 6, :, :])  # [2, 3, 16, 32, 32]
        # cls = torch.softmax(out[:, :, 7:, :, :, :], dim=2)  # [2, 3, 2, 16, 32, 32]
        
        return Δzxy, log_dwh, conf


class BBox3DHead(nn.Module):
    """3D Bounding Box prediction head with dynamic output capability"""

    def __init__(
        self,
        config: SimpleNamespace,
    ):
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_classes = config.num_classes
        self.max_bbox_len = config.max_bbox_len
        self.normalize_coords = config.normalize_coords

        self.coord_bounds = {
            "x_min": config.coord_bounds.x_min,
            "x_max": config.coord_bounds.x_max,
            "y_min": config.coord_bounds.y_min,
            "y_max": config.coord_bounds.y_max,
            "z_min": config.coord_bounds.z_min,
            "z_max": config.coord_bounds.z_max,
        }

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # 3D bbox regression: predicts normalized coordinates if enabled
        # Format: center (x,y,z) + dimensions (w,h,l) = 6 parameters (no rotation)
        self.bbox_head = nn.Linear(
            self.hidden_dim, 6 * self.max_bbox_len
        )  # 6 params per bbox

        if self.num_classes > 1:
            self.cls_head = nn.Linear(
                self.hidden_dim, self.num_classes * self.max_bbox_len
            )
        else:
            self.cls_head = None

        self.conf_head = nn.Linear(self.hidden_dim, self.max_bbox_len)

        self._init_weights()

    def _init_weights(self):
        for layer in self.feature_extractor.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)  # Small positive bias
                print(f"Initialized {layer} with Xavier weights and bias=0.1")

        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.1)

    def forward(self, x, apply_constraints=True):
        features = self.feature_extractor(x)
        bbox_pred = self.bbox_head(features).view(-1, self.max_bbox_len, 6)  # [B, N, 6]
        bbox_pred = torch.sigmoid(bbox_pred)
        return bbox_pred

    def convert_gt_to_model_format(self, gt_boxes):
        """
        Convert ground truth boxes from [x_min, y_min, z_min, x_max, y_max, z_max]
        to model format [center_x, center_y, center_z, width, height, length]

        Args:
            gt_boxes: [..., 6] - ground truth boxes in min/max format
        Returns:
            model_boxes: [..., 6] - boxes in center+size format
        """
        x_min, y_min, z_min = gt_boxes[..., 0], gt_boxes[..., 1], gt_boxes[..., 2]
        x_max, y_max, z_max = gt_boxes[..., 3], gt_boxes[..., 4], gt_boxes[..., 5]

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2

        width = x_max - x_min
        height = y_max - y_min
        length = z_max - z_min

        model_boxes = torch.stack(
            [center_x, center_y, center_z, width, height, length], dim=-1
        )

        return model_boxes

    def _apply_dynamic_filtering(self, outputs, conf_threshold=0.5):
        """
        Apply dynamic filtering based on confidence scores
        Args:
            outputs: Dictionary containing predictions
            conf_threshold: Confidence threshold for filtering
        Returns:
            Dictionary with filtered outputs
        """
        batch_size = outputs["bbox_pred"].shape[0]
        filtered_outputs = {
            "filtered_bbox_pred": [],
            "filtered_conf_pred": [],
            "filtered_cls_pred": [] if "cls_pred" in outputs else None,
            "num_valid_boxes": [],
        }

        for b in range(batch_size):
            # Get valid indices based on confidence threshold
            valid_mask = outputs["conf_pred"][b] > conf_threshold
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) > 0:
                # Filter bbox predictions
                filtered_bbox = outputs["bbox_pred"][b][valid_indices]
                filtered_conf = outputs["conf_pred"][b][valid_indices]

                filtered_outputs["filtered_bbox_pred"].append(filtered_bbox)
                filtered_outputs["filtered_conf_pred"].append(filtered_conf)

                # Filter classification predictions if available
                if "cls_pred" in outputs:
                    filtered_cls = outputs["cls_pred"][b][valid_indices]
                    filtered_outputs["filtered_cls_pred"].append(filtered_cls)

                filtered_outputs["num_valid_boxes"].append(len(valid_indices))
            else:
                # No valid boxes
                filtered_outputs["filtered_bbox_pred"].append(
                    torch.empty(0, 6, device=outputs["bbox_pred"].device)
                )
                filtered_outputs["filtered_conf_pred"].append(
                    torch.empty(0, device=outputs["conf_pred"].device)
                )

                if "cls_pred" in outputs:
                    filtered_outputs["filtered_cls_pred"].append(
                        torch.empty(
                            0, self.num_classes, device=outputs["cls_pred"].device
                        )
                    )

                filtered_outputs["num_valid_boxes"].append(0)

        return filtered_outputs

    def post_process(self, outputs, nms_threshold=0.5, max_detections=None):
        """
        Post-process predictions with NMS and top-k filtering
        Args:
            outputs: Dictionary containing model outputs
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep per batch
        Returns:
            Dictionary with post-processed outputs
        """
        if max_detections is None:
            max_detections = self.max_bbox_len

        batch_size = outputs["bbox_pred"].shape[0]
        processed_outputs = {
            "final_bbox_pred": [],
            "final_conf_pred": [],
            "final_cls_pred": [] if "cls_pred" in outputs else None,
            "final_num_boxes": [],
        }

        for b in range(batch_size):
            bbox_pred = outputs["bbox_pred"][b]  # [max_bbox_len, 6]
            conf_pred = outputs["conf_pred"][b]  # [max_bbox_len]

            # Sort by confidence
            sorted_indices = torch.argsort(conf_pred, descending=True)
            sorted_bbox = bbox_pred[sorted_indices]
            sorted_conf = conf_pred[sorted_indices]

            # Apply NMS (simplified version - you might want to use torchvision.ops.nms)
            keep_indices = self._simple_nms_3d(sorted_bbox, sorted_conf, nms_threshold)

            # Limit to max_detections
            if len(keep_indices) > max_detections:
                keep_indices = keep_indices[:max_detections]

            final_bbox = sorted_bbox[keep_indices]
            final_conf = sorted_conf[keep_indices]

            processed_outputs["final_bbox_pred"].append(final_bbox)
            processed_outputs["final_conf_pred"].append(final_conf)
            processed_outputs["final_num_boxes"].append(len(keep_indices))

            # Handle classification if available
            if "cls_pred" in outputs:
                cls_pred = outputs["cls_pred"][b][sorted_indices]
                final_cls = cls_pred[keep_indices]
                processed_outputs["final_cls_pred"].append(final_cls)

        return processed_outputs

    def _simple_nms_3d(self, boxes, scores, threshold):
        """
        Simple 3D NMS implementation for axis-aligned boxes
        Args:
            boxes: [N, 6] - center_x, center_y, center_z, width, height, length
            scores: [N] - confidence scores
            threshold: IoU threshold
        Returns:
            List of indices to keep
        """
        if len(boxes) == 0:
            return []

        keep = []
        order = torch.argsort(scores, descending=True)

        while len(order) > 0:
            i = order[0]
            keep.append(i.item())

            if len(order) == 1:
                break

            # Calculate IoU with remaining boxes
            ious = self._calculate_3d_iou(boxes[i : i + 1], boxes[order[1:]])

            # Keep boxes with IoU less than threshold
            inds = torch.where(ious <= threshold)[0]
            order = order[inds + 1]

        return keep

    def _calculate_3d_iou(self, box1, boxes2):
        """
        Calculate 3D IoU between axis-aligned boxes
        Args:
            box1: [1, 6] - center_x, center_y, center_z, width, height, length
            boxes2: [N, 6] - center_x, center_y, center_z, width, height, length
        Returns:
            IoU values [N]
        """
        # Convert center+size to min+max coordinates
        box1_min = box1[:, :3] - box1[:, 3:] / 2  # [1, 3]
        box1_max = box1[:, :3] + box1[:, 3:] / 2  # [1, 3]

        boxes2_min = boxes2[:, :3] - boxes2[:, 3:] / 2  # [N, 3]
        boxes2_max = boxes2[:, :3] + boxes2[:, 3:] / 2  # [N, 3]

        # Calculate intersection
        inter_min = torch.max(box1_min, boxes2_min)  # [N, 3]
        inter_max = torch.min(box1_max, boxes2_max)  # [N, 3]

        # Check if there's intersection
        inter_size = torch.clamp(inter_max - inter_min, min=0)  # [N, 3]
        inter_volume = inter_size.prod(dim=1)  # [N]

        # Calculate volumes
        box1_volume = box1[:, 3:].prod(dim=1)  # [1]
        boxes2_volume = boxes2[:, 3:].prod(dim=1)  # [N]

        # Calculate IoU
        union_volume = box1_volume + boxes2_volume - inter_volume
        iou = inter_volume / (union_volume + 1e-8)

        return iou

    def get_bbox_params(self, bbox_pred):
        """
        Extract individual bbox parameters (axis-aligned boxes)
        Args:
            bbox_pred: [batch_size, max_bbox_len, 6] or [max_bbox_len, 6]
        Returns:
            Dictionary with individual parameters
        """
        if bbox_pred.dim() == 2:
            bbox_pred = bbox_pred.unsqueeze(0)

        return {
            "center_x": bbox_pred[..., 0],
            "center_y": bbox_pred[..., 1],
            "center_z": bbox_pred[..., 2],
            "width": bbox_pred[..., 3],
            "height": bbox_pred[..., 4],
            "length": bbox_pred[..., 5],
        }

    def get_bbox_corners(self, bbox_pred):
        """
        Get 8 corner coordinates for axis-aligned 3D boxes
        Args:
            bbox_pred: [batch_size, max_bbox_len, 6] or [max_bbox_len, 6]
        Returns:
            corners: [..., 8, 3] - 8 corner coordinates for each box
        """
        params = self.get_bbox_params(bbox_pred)

        # Half dimensions
        hw = params["width"] / 2
        hh = params["height"] / 2
        hl = params["length"] / 2

        # 8 corners of axis-aligned box
        corners = torch.stack(
            [
                torch.stack(
                    [
                        params["center_x"] - hw,
                        params["center_y"] - hh,
                        params["center_z"] - hl,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        params["center_x"] + hw,
                        params["center_y"] - hh,
                        params["center_z"] - hl,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        params["center_x"] + hw,
                        params["center_y"] + hh,
                        params["center_z"] - hl,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        params["center_x"] - hw,
                        params["center_y"] + hh,
                        params["center_z"] - hl,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        params["center_x"] - hw,
                        params["center_y"] - hh,
                        params["center_z"] + hl,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        params["center_x"] + hw,
                        params["center_y"] - hh,
                        params["center_z"] + hl,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        params["center_x"] + hw,
                        params["center_y"] + hh,
                        params["center_z"] + hl,
                    ],
                    dim=-1,
                ),
                torch.stack(
                    [
                        params["center_x"] - hw,
                        params["center_y"] + hh,
                        params["center_z"] + hl,
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

        return corners


# Example usage with GT format conversion and normalization
if __name__ == "__main__":
    print("=== BBox Format Conversion and Normalization ===")

    # Initialize model with normalization enabled
