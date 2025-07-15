from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
coord_bounds = {
    "x_min": 0,
    "x_max": 255,
    "y_min": 0,
    "y_max": 255,
    "z_min": 0,
    "z_max": 31,
}


def denormalize_boxes(normalized_boxes, coord_bounds=coord_bounds):

    denormalized = normalized_boxes.clone()

    denormalized[..., 0] = normalized_boxes[..., 0] * (coord_bounds["x_max"])
    denormalized[..., 1] = normalized_boxes[..., 1] * coord_bounds["y_max"]
    denormalized[..., 2] = normalized_boxes[..., 2] * (coord_bounds["z_max"])
    denormalized[..., 3] = normalized_boxes[..., 3] * coord_bounds["x_max"]
    denormalized[..., 4] = normalized_boxes[..., 4] * coord_bounds["y_max"]
    denormalized[..., 5] = normalized_boxes[..., 5] * coord_bounds["z_max"]
    return denormalized

def corners_to_center(boxes):
    """
    Convert bounding boxes from corner coordinates to center format.
    
    Args:
        boxes: Tensor of shape (..., 6) where last dim is (x_min, y_min, z_min, x_max, y_max, z_max)
    
    Returns:
        Tensor of shape (..., 6) where last dim is (cx, cy, cz, width, height, depth)
    """
    # Calculate center coordinates
    cx = (boxes[..., 0] + boxes[..., 3]) / 2
    cy = (boxes[..., 1] + boxes[..., 4]) / 2
    cz = (boxes[..., 2] + boxes[..., 5]) / 2
    
    # Calculate dimensions
    width = boxes[..., 3] - boxes[..., 0]
    height = boxes[..., 4] - boxes[..., 1]
    depth = boxes[..., 5] - boxes[..., 2]
    
    return torch.stack([cx, cy, cz, width, height, depth], dim=-1)

def convert_model_to_gt_format(pred_boxes, normalize_coords=True):
    """
    Convert model predictions from [center_x, center_y, center_z, width, height, length]
    to ground truth format [x_min, y_min, z_min, x_max, y_max, z_max] with safe clamping.

    Args:
        pred_box_boxes: [..., 6] - boxes in center+size format
        normalize_coords: If True, ensures output stays in [0,1] range
    Returns:
        gt_boxes: [..., 6] - boxes in min/max format
    """
    if isinstance(pred_boxes, list):
        pred_boxes = torch.tensor(pred_boxes)
    
    if pred_boxes.dim() == 1:
        pred_boxes = pred_boxes.unsqueeze(0)
    
    boxes = pred_boxes.clone()
    
    
    # Step 1: Apply activation to height (if using log-scale)
    # boxes[:, 4] = torch.exp(boxes[:, 4])  # Uncomment if predictions are log-scale
    boxes[:, 4] = torch.abs(boxes[:, 4])  # Ensure height is non-zero
    
    # Step 2: Convert center-size to min-max
    x_min = boxes[:, 0] - boxes[:, 3] / 2
    y_min = boxes[:, 1] - boxes[:, 4] / 2  # Critical: Use predicted height
    z_min = boxes[:, 2] - boxes[:, 5] / 2
    x_max = boxes[:, 0] + boxes[:, 3] / 2
    y_max = boxes[:, 1] + boxes[:, 4] / 2
    z_max = boxes[:, 2] + boxes[:, 5] / 2
    
    x_min = torch.clamp(x_min, min=0)
    y_min = torch.clamp(y_min, min=0)
    z_min = torch.clamp(z_min, min=0)
    x_max = torch.clamp(x_max, max=coord_bounds["x_max"])  # W
    y_max = torch.clamp(y_max, max=coord_bounds["y_max"])  # H
    z_max = torch.clamp(z_max, max=coord_bounds["z_max"])  # D
    
    minmax_boxes = torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)
    
    
    return minmax_boxes
    # Extract center and dimensions (non-inplace)
    # centers = model_boxes[..., :3]  # [..., 3]
    # dimensions = model_boxes[..., 3:]  # [..., 3]

    # # Apply constraints safely
    # if normalize_coords:
    #     # Clamp centers and ensure positive dimensions
    #     centers = torch.clamp(centers, 0.0, 1.0)
    #     dimensions = torch.clamp(dimensions, min=1e-6)  # Avoid zero dimensions

    # # Convert to min/max coordinates
    # half_dims = dimensions / 2
    # min_coords = centers - half_dims
    # max_coords = centers + half_dims

    # # Final clamping to ensure all values in [0,1] when normalized
    # if normalize_coords:
    #     min_coords = torch.clamp(min_coords, 0.0, 1.0)
    #     max_coords = torch.clamp(max_coords, 0.0, 1.0)

    # # Stack into GT format
    # gt_boxes = torch.cat([min_coords, max_coords], dim=-1)

    # return gt_boxes


def compute_ious(
    #
    bbox_preds,
    targets,
    masks,
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

    matches = hungarian_iou_matching(bbox_preds, gt_boxes_minmax)
    for pred_idx, gt_idx, iou in matches:
        pred_box = bbox_preds[pred_idx]
        gt_box = gt_boxes_minmax[gt_idx]
        abs_iou = box3d_iou_single(pred_box, gt_box)
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

def iou_loss(bbox1,bbox2,denormalize=False):
    return 1-box3d_iou_single(bbox1,bbox2,denormalize=denormalize)

def box3d_iou_single(bbox1, bbox2, denormalize=False):
    """
    Compute 3D IoU between two 3D bounding boxes in min-max format (x_min, y_min, z_min, x_max, y_max, z_max)

    Args:
        box1: [6] - first 3D bounding box
        box2: [6] - second 3D bounding box
        denormalize: If True, denormalizes box coordinates from [0,1] to [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        iou: IoU value between the two boxes
    """
    if denormalize:
        box1 = denormalize_boxes(bbox1)
        box2 = denormalize_boxes(bbox2)
    # print("box1", box1.tolist(), "box2", box2.tolist())
    box1 = box1.squeeze()  # Converts [1,6] → [6]
    box2 = box2.squeeze()  # Converts [1,6] → [6]
    inter_min = torch.max(box1[:3], box2[:3])
    inter_max = torch.min(box1[3:], box2[3:])
    inter_dim = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_dim.prod()
    # print("inter_min", inter_min, "inter_max", inter_max, "inter_dim", inter_dim, "inter_vol", inter_vol)
    # Volumes
    vol1 = (box1[3:] - box1[:3]).prod()
    vol2 = (box2[3:] - box2[:3]).prod()

    union_vol = vol1 + vol2 - inter_vol + 1e-8
    iou = inter_vol / union_vol
    # print("union_vol", union_vol, "iou", iou)
    return iou

def compute_3d_volume(boxes):
    """
    Compute volume of 3D boxes.
    Args:
        boxes: (..., 6) tensor where last dim is (z,x,y,d,w,h)
    Returns:
        volumes: (...) tensor of box volumes
    """
    d, w, h = boxes[..., 3], boxes[..., 4], boxes[..., 5]
    return d * w * h

def compute_diagonal_length(boxes):
    """
    Compute diagonal length of 3D boxes.
    Args:
        boxes: (..., 6) tensor where last dim is (z,x,y,d,w,h)
    Returns:
        diags: (...) tensor of diagonal lengths
    """
    d, w, h = boxes[..., 3], boxes[..., 4], boxes[..., 5]
    return torch.sqrt(d**2 + w**2 + h**2)

def compute_3d_intersection(pred_boxes, gt_boxes):
    """
    Compute intersection volume between predicted and ground truth 3D boxes.
    Args:
        pred_boxes: (B, N, 6) - (z,x,y,d,w,h)
        gt_boxes: (B, N, 6) - (z,x,y,d,w,h)
    Returns:
        intersection: (B, N) tensor of intersection volumes
    """
    # Convert to corner format (min_z, min_x, min_y, max_z, max_x, max_y)
    def get_corners(boxes):
        z, x, y, d, w, h = boxes.unbind(-1)
        half_d, half_w, half_h = d/2, w/2, h/2
        min_z, max_z = z - half_d, z + half_d
        min_x, max_x = x - half_w, x + half_w
        min_y, max_y = y - half_h, y + half_h
        return torch.stack([min_z, min_x, min_y, max_z, max_x, max_y], dim=-1)
    
    pred_corners = get_corners(pred_boxes)  # (B, N, 6)
    gt_corners = get_corners(gt_boxes)      # (B, N, 6)
    
    # Compute intersection for each box pair
    min_z = torch.max(pred_corners[..., 0], gt_corners[..., 0])
    min_x = torch.max(pred_corners[..., 1], gt_corners[..., 1])
    min_y = torch.max(pred_corners[..., 2], gt_corners[..., 2])
    
    max_z = torch.min(pred_corners[..., 3], gt_corners[..., 3])
    max_x = torch.min(pred_corners[..., 4], gt_corners[..., 4])
    max_y = torch.min(pred_corners[..., 5], gt_corners[..., 5])
    
    # Clip to zero if no intersection
    inter_d = torch.clamp(max_z - min_z, min=0)
    inter_w = torch.clamp(max_x - min_x, min=0)
    inter_h = torch.clamp(max_y - min_y, min=0)
    
    return inter_d * inter_w * inter_h  # (B, N)

def standard_loss(bbox_preds, masks, targets, neg_weight=1.0, lambda_reg=1.0):
        total_loss=[]
        valid_batches = 0
        
        ious=[]
        for b in range(len(bbox_preds)):
            bbox_pred = bbox_preds[b]  # [num_preds, 6]
            mask = masks[b]  # [max_num_gt]
            target = targets[b]  # [max_num_gt, 6]
            gt_boxes = target[mask]  # [num_valid_gt, 6]
            if len(gt_boxes) == 0:
                continue
            gt_boxes = gt_boxes[:1]  # [1, 6]
            mask = mask[:1]  # [1]
            # Convert format if needed
            # print("before convert_model_to_gt_format bbox_pred", bbox_pred.tolist())
            pred_boxes_minmax = convert_model_to_gt_format(bbox_pred)
            # print("pred_boxes_minmax ", pred_boxes_minmax.tolist())
            # print("gt_boxes ", gt_boxes.tolist())
            # batch_loss = F.smooth_l1_loss(
            #     pred_boxes_minmax,
            #     gt_boxes,
            #     reduction='sum'  # Preserve magnitude
            # )
            mse_weight = 1.0
            iou_weight = 2.0  # IoU loss is often more important
            l2_weight = 0.0001  # Much smaller regularization
            print("pred_boxes_minmax shape", pred_boxes_minmax, "gt_boxes shape", gt_boxes)
            loss_iou= iou_loss(pred_boxes_minmax, gt_boxes, denormalize=True)
            print("bbox iou", 1-loss_iou)
            loss = mse_weight * F.mse_loss(pred_boxes_minmax, gt_boxes) + \
                iou_weight * loss_iou + \
                l2_weight * torch.sum(pred_boxes_minmax ** 2)
            # l2_reg = 0.01 * torch.sum(pred_boxes_minmax ** 2)
    
            # loss = F.mse_loss(pred_boxes_minmax, gt_boxes) + l2_reg
            # iou = box3d_iou_single(pred_boxes_minmax, gt_boxes, denormalize=True)

            # total_loss.append( (loss_iou+ loss) )
            total_loss.append(loss_iou + 0.1 * loss)
        return torch.stack(total_loss).mean() if total_loss else torch.tensor(0.0), 0.0
def diou_3d(pred_boxes, gt_boxes):
    """
    3D Distance-IoU Loss (DIoU)
    Args:
        pred_boxes: (B, N, 6) - (z,x,y,d,w,h)
        gt_boxes: (B, N, 6) - (z,x,y,d,w,h)
    Returns:
        diou_loss: scalar tensor
    """
    # Compute 3D IoU
    intersection = compute_3d_intersection(pred_boxes, gt_boxes)  # (B, N)
    pred_vol = compute_3d_volume(pred_boxes)  # (B, N)
    gt_vol = compute_3d_volume(gt_boxes)      # (B, N)
    union = pred_vol + gt_vol - intersection
    iou = intersection / (union + 1e-7)  # (B, N)
    
    # Compute center distance
    pred_centers = pred_boxes[..., :3]  # (B, N, 3)
    gt_centers = gt_boxes[..., :3]      # (B, N, 3)
    c_dist = torch.norm(pred_centers - gt_centers, dim=-1)  # (B, N)
    
    # Compute diagonal distance of GT boxes
    diag_dist = compute_diagonal_length(gt_boxes)  # (B, N)
    
    # DIoU = IoU - (center_dist / diagonal_dist)
    diou = iou - (c_dist / (diag_dist + 1e-7))  # (B, N)
    
    # Loss = 1 - DIoU
    return 1 - diou.mean()  # Scalar


if __name__ == "__main__":
    # box1=torch.tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    box_pred = torch.tensor([0.3401879072189331, 0.21133515238761902, 0.32378238439559937, 0.8024861812591553, 0.7225353717803955, 0.7604655623435974])
    box_gt = torch.tensor([0.3231697678565979, 0.20672544836997986, 0.0010000000474974513, 0.548652172088623, 0.4423169493675232, 0.96875])
    iou = box3d_iou_single(box_pred, box_gt, denormalize=True)
    print("iou", iou)
