from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import torch

coord_bounds = {
    "x_min": 0,
    "x_max": 255,
    "y_min": 0,
    "y_max": 255,
    "z_min": 0,
    "z_max": 31,
}


def denormalize_boxes(normalized_boxes, coord_bounds=coord_bounds):
    """
    Denormalize boxes from [0, 1] range to original coordinates
    Args:
        normalized_boxes: [..., 6] - normalized boxes
    Returns:
        boxes: [..., 6] - denormalized boxes
    """

    denormalized = normalized_boxes.clone()

    # Denormalize centers
    x_range = coord_bounds["x_max"] - coord_bounds["x_min"]
    y_range = coord_bounds["y_max"] - coord_bounds["y_min"]
    z_range = coord_bounds["z_max"] - coord_bounds["z_min"]

    denormalized[..., 0] = (
        normalized_boxes[..., 0] * x_range + coord_bounds["x_min"]
    )  # center_x
    denormalized[..., 1] = (
        normalized_boxes[..., 1] * y_range + coord_bounds["y_min"]
    )  # center_y
    denormalized[..., 2] = (
        normalized_boxes[..., 2] * z_range + coord_bounds["z_min"]
    )  # center_z

    # Denormalize dimensions
    denormalized[..., 3] = normalized_boxes[..., 3] * x_range  # width
    denormalized[..., 4] = normalized_boxes[..., 4] * y_range  # height
    denormalized[..., 5] = normalized_boxes[..., 5] * z_range  # length
    return denormalized


def convert_model_to_gt_format(model_boxes, normalize_coords=True):
    """
    Convert model predictions from [center_x, center_y, center_z, width, height, length]
    to ground truth format [x_min, y_min, z_min, x_max, y_max, z_max]

    Args:
        model_boxes: [..., 6] - boxes in center+size format
    Returns:
        gt_boxes: [..., 6] - boxes in min/max format
    """
    # Denormalize if needed
    if normalize_coords:
        model_boxes = denormalize_boxes(model_boxes)

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


def compute_ious(
    #
    bbox_preds,
    targets,
    masks,
    denormalize_gt=True,
):
    """
    Compute IoU matrix between predicted and ground truth 3D boxes
    Args:
        pred_boxes: [N, 6] in format [x_min, y_min, z_min, x_max, y_max, z_max]
        gt_boxes:   [M, 6] in same format
    Returns:
        iou_matrix: [N, M] IoU values
    """

    pairs = defaultdict(list)
    gt_boxes_minmax = targets[masks]
    if len(gt_boxes_minmax) == 0:
        return
    # if denormalize_gt:
    #     gt_boxes_min_max=denormalize_boxes(gt_boxes_minmax)
    print("gt  bbox", gt_boxes_minmax)
    print("before", bbox_preds)
    bbox_pred_minmax = convert_model_to_gt_format(bbox_preds, normalize_coords=False)
    print("after", bbox_pred_minmax)
    matches = hungarian_iou_matching(bbox_pred_minmax, gt_boxes_minmax)
    print("matches", matches)
    for pred_idx, gt_idx, iou in matches:
        pred_box = bbox_preds[pred_idx]
        gt_box = gt_boxes_minmax[gt_idx]
        abs_iou = box3d_iou_single(pred_box, gt_box, denormalize=denormalize_boxes)
    pairs["pred"].append(pred_box)
    pairs["gt"].append(gt_box)
    pairs["iou"].append(abs_iou)
    return pairs


def hungarian_iou_matching(pred_boxes, gt_boxes):
    """
    Matches predicted boxes to GT boxes using Hungarian algorithm based on IoU.

    Args:
        pred_boxes (Tensor): [N, 6] boxes in min-max format (x_min, y_min, z_min, x_max, y_max, z_max)
        gt_boxes (Tensor): [M, 6] ground truth boxes in min-max format

    Returns:
        matches: List of (pred_idx, gt_idx, iou_score)
    """
    device = pred_boxes.device
    N, M = pred_boxes.size(0), gt_boxes.size(0)
    if N == 0 or M == 0:
        return []

    # Compute IoU matrix: [N, M]
    iou_matrix = compute_3d_iou_matrix(pred_boxes, gt_boxes)  # assumed implemented
    cost_matrix = (
        1.0 - iou_matrix.detach().cpu().numpy()
    )  # Cost = 1 - IoU (lower is better)

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        iou = iou_matrix[r, c].item()
        matches.append((r, c, iou))

    return matches


def compute_3d_iou_matrix(pred_boxes, gt_boxes):
    """
    Compute IoU matrix between predicted and ground truth 3D boxes
    Args:
        pred_boxes: [N, 6] in format [x_min, y_min, z_min, x_max, y_max, z_max]
        gt_boxes:   [M, 6] in same format
    Returns:
        iou_matrix: [N, M] IoU values
    """
    N = pred_boxes.size(0)
    M = gt_boxes.size(0)

    iou_matrix = torch.zeros(N, M, device=pred_boxes.device)

    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = box3d_iou_single(pred_boxes[i], gt_boxes[j])

    return iou_matrix


def box3d_iou_single(box1, box2, denormalize=None):
    if denormalize:
        box1 = denormalize(box1)
        box2 = denormalize(box2)
        # print("box1",box1,"box2",box2)
    inter_min = torch.max(box1[:3], box2[:3])
    inter_max = torch.min(box1[3:], box2[3:])
    inter_dim = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_dim.prod()

    # Volumes
    vol1 = (box1[3:] - box1[:3]).prod()
    vol2 = (box2[3:] - box2[:3]).prod()

    union_vol = vol1 + vol2 - inter_vol + 1e-8
    iou = inter_vol / union_vol
    return iou
