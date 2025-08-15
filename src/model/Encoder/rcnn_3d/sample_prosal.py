import torch
from torchvision.ops import nms
import torch.nn.functional as F
import numpy as np

def encode_box_targets_3d(anchors, gt_boxes):
    """
    anchors: (N, 6) [z_center, y_center, x_center, depth, height, width]
    gt_boxes: (N, 6) same format as anchors
    """
    # Centers
    za, ya, xa, da, ha, wa = anchors.unbind(dim=1)
    zg, yg, xg, dg, hg, wg = gt_boxes.unbind(dim=1)

    # Deltas
    t_z = (zg - za) / da
    t_y = (yg - ya) / ha
    t_x = (xg - xa) / wa

    t_d = torch.log(dg / da)
    t_h = torch.log(hg / ha)
    t_w = torch.log(wg / wa)

    return torch.stack((t_z, t_y, t_x, t_d, t_h, t_w), dim=1)

def compute_rpn_loss(pred_objectness, pred_bbox_deltas, anchors, gt_boxes, iou_threshold_pos=0.5, iou_threshold_neg=0.1):
    """
    Args:
        pred_objectness: [N, 2] logits for objectness (fg/bg)
        pred_bbox_deltas: [N, 6] predicted box deltas
        anchors: [N, 6] anchor boxes (z1,y1,x1,z2,y2,x2)
        gt_boxes: [M, 6] ground truth boxes
    Returns:
        total_loss: scalar tensor
    """
    device = pred_objectness.device
    num_anchors = anchors.size(0)

    # 1. Assign labels using IoU
    ious = box_iou_3d(anchors, gt_boxes)
    max_iou_per_anchor, gt_assignment = ious.max(dim=1)

    labels = torch.full((num_anchors,), -1, dtype=torch.int64, device=device)
    labels[max_iou_per_anchor >= iou_threshold_pos] = 1
    labels[max_iou_per_anchor < iou_threshold_neg] = 0

    # 2. Objectness loss
    objectness_loss = F.cross_entropy(pred_objectness, labels.clamp(min=0))

    # 3. Box regression loss (only on positives)
    positive_mask = labels == 1
    if positive_mask.any():
        target_deltas = encode_box_targets_3d(anchors[positive_mask], gt_boxes[gt_assignment[positive_mask]])
        box_loss = F.smooth_l1_loss(pred_bbox_deltas[positive_mask], target_deltas)
    else:
        box_loss = torch.tensor(0.0, device=device)

    return objectness_loss + box_loss

def box_iou_3d(boxes1, boxes2):
    """
    Compute IoU between two sets of 3D boxes.
    boxes1: [N, 6]  (z1, y1, x1, z2, y2, x2)
    boxes2: [M, 6]  (z1, y1, x1, z2, y2, x2)
    Returns:
        ious: [N, M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), device=boxes1.device)

    # Ensure proper order
    b1_min = boxes1[:, 0:3]
    b1_max = boxes1[:, 3:6]
    b2_min = boxes2[:, 0:3]
    b2_max = boxes2[:, 3:6]

    # Compute intersection corners
    inter_min = torch.max(b1_min[:, None, :], b2_min[None, :, :])
    inter_max = torch.min(b1_max[:, None, :], b2_max[None, :, :])
    inter_dims = torch.clamp(inter_max - inter_min, min=0)
    inter_vol = inter_dims.prod(dim=2)

    # Volumes of boxes
    vol1 = (b1_max - b1_min).prod(dim=1)
    vol2 = (b2_max - b2_min).prod(dim=1)

    # IoU
    union_vol = vol1[:, None] + vol2[None, :] - inter_vol
    iou = inter_vol / torch.clamp(union_vol, min=1e-6)
    return iou

def _sample_proposals( proposals, targets=None, num_samples=64,training=True):
    """
    proposals: Tensor [P, 6]  # (z1, y1, x1, z2, y2, x2)
    targets: list[dict] or None, each dict with keys: 'boxes' [N, 6], 'labels' [N]
    num_samples: number of RoIs to keep after sampling
    """

    device = proposals.device

    if training:
        if targets is None:
            raise ValueError("Targets must be provided during training for proposal sampling.")

        # ---- Match proposals to ground truth boxes ----
        matched_idxs = []
        labels_list = []
        for i, tgt in enumerate(targets):
            gt_boxes = tgt["boxes"]  # [N, 6]
            gt_labels = tgt["labels"]  # [N]
            # Compute IoU between proposals and gt_boxes
            ious = box_iou_3d(proposals, gt_boxes)  # custom IoU for 3D boxes
            max_iou, argmax_iou = ious.max(dim=1)

            # Assign labels: positive if IoU > 0.5, negative if < 0.3, ignore otherwise
            labels = torch.full((proposals.size(0),), fill_value=-1, dtype=torch.int64, device=device)
            labels[max_iou < 0.3] = 0
            labels[max_iou >= 0.5] = gt_labels[argmax_iou[max_iou >= 0.5]]

            matched_idxs.append(argmax_iou)
            labels_list.append(labels)

        labels = torch.cat(labels_list, dim=0)

        # ---- Subsample: balance positive/negative ----
        positive_idx = torch.where(labels > 0)[0]
        negative_idx = torch.where(labels == 0)[0]

        num_pos = min(positive_idx.numel(), num_samples // 2)
        num_neg = num_samples - num_pos

        perm_pos = torch.randperm(positive_idx.numel(), device=device)[:num_pos]
        perm_neg = torch.randperm(negative_idx.numel(), device=device)[:num_neg]

        keep_idx = torch.cat([positive_idx[perm_pos], negative_idx[perm_neg]], dim=0)

        rois = proposals[keep_idx]
        roi_labels = labels[keep_idx]

        rpn_loss = compute_rpn_loss(proposals, targets)  # You can replace with your own

        return rois, roi_labels, rpn_loss

    else:
        # ==== Validation / Inference ====
        # Just return proposals with no labels and zero RPN loss
        return proposals, None, torch.tensor(0.0, device=device)