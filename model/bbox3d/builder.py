import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

def build_bbox3d_module():
    """
    Build the 3D Bounding Box prediction module based on configuration
    Args:
        config: Configuration object containing model parameters
    Returns:
        BBox3DHead instance
    """
 
    return BBox3DHead()


class BBox3DHead(nn.Module):
    """3D Bounding Box prediction head with dynamic output capability"""

    def __init__(
        self,
        input_dim: int=512,
        hidden_dim: int = 512,
        num_classes: int = 1,
        max_bbox_len: int = 9,
        normalize_coords: bool = True,
        coord_bounds: dict = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_bbox_len = max_bbox_len
        self.normalize_coords = normalize_coords

        # Default coordinate bounds for normalization
        if coord_bounds is None:
            self.coord_bounds = {
                "x_min": -10.0,
                "x_max": 10.0,
                "y_min": -10.0,
                "y_max": 10.0,
                "z_min": -5.0,
                "z_max": 5.0,
            }
        else:
            self.coord_bounds = coord_bounds

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # 3D bbox regression: predicts normalized coordinates if enabled
        # Format: center (x,y,z) + dimensions (w,h,l) = 6 parameters (no rotation)
        self.bbox_head = nn.Linear(hidden_dim, 6 * max_bbox_len)  # 6 params per bbox

        # Classification head for each bbox
        if num_classes > 1:
            self.cls_head = nn.Linear(hidden_dim, num_classes * max_bbox_len)
        else:
            self.cls_head = None

        # Confidence/objectness head for each bbox
        self.conf_head = nn.Linear(hidden_dim, max_bbox_len)

        # self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x, dynamic_output=True, conf_threshold=0.5, apply_constraints=True
    ):
        # torch.Size([2, 6144])
        """
        Forward pass with dynamic output capability
        Args:
            x: Input features [batch_size, input_dim]
            dynamic_output: Whether to filter outputs based on confidence
            conf_threshold: Confidence threshold for dynamic filtering
            apply_constraints: Whether to apply bbox constraints (positive dimensions)
        Returns:
            Dict containing bbox predictions, confidence, and optionally classification
        """
        batch_size = x.shape[0]
        features = self.feature_extractor(x)

        # Predict all possible bboxes
        bbox_pred = self.bbox_head(features)  # [batch_size, 6 * max_bbox_len]
        bbox_pred = bbox_pred.view(
            batch_size, self.max_bbox_len, 6
        )  # [batch_size, max_bbox_len, 6]

        # Apply constraints to bbox predictions
        # if apply_constraints:
        #     bbox_pred = self._apply_bbox_constraints(bbox_pred)

        # Confidence scores for each bbox
        conf_pred = torch.sigmoid(
            self.conf_head(features)
        )  # [batch_size, max_bbox_len]

        outputs = {
            "bbox_pred": bbox_pred,
            "conf_pred": conf_pred,
            "raw_bbox_pred": (
                bbox_pred if not apply_constraints else None
            ),  # Keep raw predictions for debugging
        }

        # Classification predictions if multi-class
        if self.cls_head is not None:
            cls_pred = self.cls_head(
                features
            )  # [batch_size, num_classes * max_bbox_len]
            cls_pred = cls_pred.view(batch_size, self.max_bbox_len, self.num_classes)
            outputs["cls_pred"] = cls_pred

        # Dynamic output filtering
        if dynamic_output:
            filtered_outputs = self._apply_dynamic_filtering(outputs, conf_threshold)
            outputs.update(filtered_outputs)

        return outputs

    def convert_gt_to_model_format(self, gt_boxes):
        """
        Convert ground truth boxes from [x_min, y_min, z_min, x_max, y_max, z_max]
        to model format [center_x, center_y, center_z, width, height, length]

        Args:
            gt_boxes: [..., 6] - ground truth boxes in min/max format
        Returns:
            model_boxes: [..., 6] - boxes in center+size format
        """
        # Extract min/max coordinates
        x_min, y_min, z_min = gt_boxes[..., 0], gt_boxes[..., 1], gt_boxes[..., 2]
        x_max, y_max, z_max = gt_boxes[..., 3], gt_boxes[..., 4], gt_boxes[..., 5]

        # Convert to center + dimensions
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2

        width = x_max - x_min
        height = y_max - y_min
        length = z_max - z_min

        # Stack into model format
        model_boxes = torch.stack(
            [center_x, center_y, center_z, width, height, length], dim=-1
        )

        # Normalize if enabled
        if self.normalize_coords:
            model_boxes = self.normalize_boxes(model_boxes)

        return model_boxes

    def convert_model_to_gt_format(self, model_boxes):
        """
        Convert model predictions from [center_x, center_y, center_z, width, height, length]
        to ground truth format [x_min, y_min, z_min, x_max, y_max, z_max]

        Args:
            model_boxes: [..., 6] - boxes in center+size format
        Returns:
            gt_boxes: [..., 6] - boxes in min/max format
        """
        # Denormalize if needed
        if self.normalize_coords:
            model_boxes = self.denormalize_boxes(model_boxes)

        # Extract center and dimensions
        center_x, center_y, center_z = (
            model_boxes[..., 0],
            model_boxes[..., 1],
            model_boxes[..., 2],
        )
        width, height, length = (
            model_boxes[..., 3],
            model_boxes[..., 4],
            model_boxes[..., 5],
        )

        # Convert to min/max coordinates
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        z_min = center_z - length / 2

        x_max = center_x + width / 2
        y_max = center_y + height / 2
        z_max = center_z + length / 2

        # Stack into GT format
        gt_boxes = torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=-1)

        return gt_boxes

    def normalize_boxes(self, boxes):
        """
        Normalize boxes to [0, 1] range
        Args:
            boxes: [..., 6] - [center_x, center_y, center_z, width, height, length]
        Returns:
            normalized_boxes: [..., 6] - normalized to [0, 1]
        """
        normalized = boxes.clone()

        # Normalize centers to [0, 1]
        x_range = self.coord_bounds["x_max"] - self.coord_bounds["x_min"]
        y_range = self.coord_bounds["y_max"] - self.coord_bounds["y_min"]
        z_range = self.coord_bounds["z_max"] - self.coord_bounds["z_min"]

        normalized[..., 0] = (
            boxes[..., 0] - self.coord_bounds["x_min"]
        ) / x_range  # center_x
        normalized[..., 1] = (
            boxes[..., 1] - self.coord_bounds["y_min"]
        ) / y_range  # center_y
        normalized[..., 2] = (
            boxes[..., 2] - self.coord_bounds["z_min"]
        ) / z_range  # center_z

        # Normalize dimensions by the respective ranges
        normalized[..., 3] = boxes[..., 3] / x_range  # width
        normalized[..., 4] = boxes[..., 4] / y_range  # height
        normalized[..., 5] = boxes[..., 5] / z_range  # length

        return normalized

    def denormalize_boxes(self, normalized_boxes):
        """
        Denormalize boxes from [0, 1] range to original coordinates
        Args:
            normalized_boxes: [..., 6] - normalized boxes
        Returns:
            boxes: [..., 6] - denormalized boxes
        """
        denormalized = normalized_boxes.clone()

        # Denormalize centers
        x_range = self.coord_bounds["x_max"] - self.coord_bounds["x_min"]
        y_range = self.coord_bounds["y_max"] - self.coord_bounds["y_min"]
        z_range = self.coord_bounds["z_max"] - self.coord_bounds["z_min"]

        denormalized[..., 0] = (
            normalized_boxes[..., 0] * x_range + self.coord_bounds["x_min"]
        )  # center_x
        denormalized[..., 1] = (
            normalized_boxes[..., 1] * y_range + self.coord_bounds["y_min"]
        )  # center_y
        denormalized[..., 2] = (
            normalized_boxes[..., 2] * z_range + self.coord_bounds["z_min"]
        )  # center_z

        # Denormalize dimensions
        denormalized[..., 3] = normalized_boxes[..., 3] * x_range  # width
        denormalized[..., 4] = normalized_boxes[..., 4] * y_range  # height
        denormalized[..., 5] = normalized_boxes[..., 5] * z_range  # length

        return denormalized
        """
        Apply constraints to bbox predictions to ensure valid boxes
        Args:
            bbox_pred: [batch_size, max_bbox_len, 6] - center_x, center_y, center_z, width, height, length
        Returns:
            Constrained bbox predictions
        """
        constrained_bbox = bbox_pred.clone()

        # Centers can be any value (positive or negative depending on coordinate system)
        # No constraints on center coordinates: constrained_bbox[..., :3] = bbox_pred[..., :3]

        # Dimensions (width, height, length) must be positive
        # Option 1: Use ReLU to ensure positive values
        constrained_bbox[..., 3:] = (
            F.relu(bbox_pred[..., 3:]) + 1e-6
        )  # Add small epsilon to avoid zero

        # Option 2: Use exponential to ensure positive values (alternative)
        # constrained_bbox[..., 3:] = torch.exp(bbox_pred[..., 3:])

        # Option 3: Use softplus for smooth positive constraint (alternative)
        # constrained_bbox[..., 3:] = F.softplus(bbox_pred[..., 3:])

        return constrained_bbox

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
    coord_bounds = {
        "x_min": -10.0,
        "x_max": 10.0,
        "y_min": -10.0,
        "y_max": 10.0,
        "z_min": -5.0,
        "z_max": 5.0,
    }

    model = BBox3DHead(
        input_dim=6144,
        hidden_dim=512,
        num_classes=10,
        max_bbox_len=9,
        normalize_coords=True,
        coord_bounds=coord_bounds,
    )
    inp=torch.rand(2,6144)
    output=model(inp)
    print("output",output)

    # # Example ground truth boxes in your format: [x_min, y_min, z_min, x_max, y_max, z_max]
    # gt_boxes_minmax = torch.tensor(
    #     [
    #         [1.0, 2.0, 0.5, 4.0, 5.0, 2.0],  # Box 1
    #         [-2.0, -1.0, -0.5, 1.0, 2.0, 1.5],  # Box 2
    #         [0.0, 0.0, 0.0, 3.0, 3.0, 3.0],  # Box 3
    #     ]
    # )

    # print(f"Ground Truth (min/max format): {gt_boxes_minmax}")

    # # Convert to model format (center + size)
    # gt_boxes_center = model.convert_gt_to_model_format(gt_boxes_minmax)
    # print(f"Ground Truth (center+size format): {gt_boxes_center}")
    # print(f"Normalized: {model.normalize_coords}")

    # # Example model predictions
    # batch_size = 2
    # input_features = torch.randn(batch_size, 256)

    # # Forward pass
    # outputs = model(input_features, apply_constraints=True)
    # print(f"\nModel predictions shape: {outputs['bbox_pred'].shape}")
    # print(f"Sample prediction (normalized): {outputs['bbox_pred'][0, 0]}")

    # # Convert model predictions back to GT format
    # pred_gt_format = model.convert_model_to_gt_format(
    #     outputs["bbox_pred"][0:1, 0:1]
    # )  # First box
    # print(f"Prediction in GT format: {pred_gt_format}")

    # print(f"\n=== Benefits of Normalization ===")
    # print("✅ Stable training (values in [0,1] range)")
    # print("✅ Better gradient flow")
    # print("✅ Easier to set learning rates")
    # print("✅ Model learns relative positions/sizes")
    # print("✅ Works well with sigmoid activation")

    # print(f"\n=== Training Pipeline ===")
    # print("1. Load GT boxes: [x_min, y_min, z_min, x_max, y_max, z_max]")
    # print(
    #     "2. Convert to model format: [center_x, center_y, center_z, width, height, length]"
    # )
    # print("3. Normalize to [0,1]: Apply coordinate bounds")
    # print("4. Train model with normalized targets")
    # print("5. Apply sigmoid constraints during inference")
    # print("6. Denormalize predictions for evaluation")

    # # Example loss calculation setup
    # print(f"\n=== Loss Calculation Example ===")
    # # You would use gt_boxes_center as targets for training
    # print(f"Targets for training: {gt_boxes_center}")
    # print("Loss = MSE(model_predictions, normalized_gt_boxes)")
    # print("+ confidence_loss + classification_loss")
