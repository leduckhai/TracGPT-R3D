import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ================================
# Segmentation Loss Classes
# ================================

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        target_ = target.clone().float()
        target_[target == -1] = 0
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match\n" + str(predict.shape) + '\n' + str(target.shape[0])
        predict = predict.contiguous().view(predict.shape[0], -1)
        target_ = target_.contiguous().view(target_.shape[0], -1)

        num = torch.sum(torch.mul(predict, target_), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target_, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        # dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]
        dice_loss_avg = dice_loss.sum() / dice_loss.shape[0]

        return dice_loss_avg


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match\n' + str(predict.shape) + '\n' + str(target.shape)
        target_ = target.clone()
        target_[target == -1] = 0

        ce_loss = self.criterion(predict, target_.float())

        return ce_loss


# ================================
# 3D Bounding Box Loss Classes
# ================================

class L1Loss(nn.Module):
    """L1 Loss for 3D bounding box regression"""
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, predict, target):
        """
        Args:
            predict: Predicted 3D bounding boxes [B, N, 7] (x, y, z, w, h, l, r)
            target: Ground truth 3D bounding boxes [B, N, 7] (x, y, z, w, h, l, r)
        """
        assert predict.shape == target.shape, f'predict & target shape do not match: {predict.shape} vs {target.shape}'
        
        # Handle invalid targets (marked with -1)
        valid_mask = (target != -1).all(dim=-1)  # [B, N]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predict.device, requires_grad=True)
        
        # Apply mask to get valid predictions and targets
        valid_predict = predict[valid_mask]  # [valid_boxes, 7]
        valid_target = target[valid_mask]    # [valid_boxes, 7]
        
        l1_loss = self.criterion(valid_predict, valid_target)
        
        return l1_loss


class IoU3DLoss(nn.Module):
    """3D IoU Loss for 3D bounding box regression"""
    def __init__(self, reduction='mean', eps=1e-7):
        super(IoU3DLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, predict, target):
        """
        Args:
            predict: Predicted 3D bounding boxes [B, N, 7] (x, y, z, w, h, l, r)
            target: Ground truth 3D bounding boxes [B, N, 7] (x, y, z, w, h, l, r)
        """
        assert predict.shape == target.shape, f'predict & target shape do not match: {predict.shape} vs {target.shape}'
        
        # Handle invalid targets (marked with -1)
        valid_mask = (target != -1).all(dim=-1)  # [B, N]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predict.device, requires_grad=True)
        
        # Apply mask to get valid predictions and targets
        valid_predict = predict[valid_mask]  # [valid_boxes, 7]
        valid_target = target[valid_mask]    # [valid_boxes, 7]
        
        # Calculate 3D IoU
        iou_3d = self.calculate_3d_iou(valid_predict, valid_target)
        
        # Convert IoU to loss (1 - IoU)
        iou_loss = 1.0 - iou_3d
        
        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            return iou_loss

    def calculate_3d_iou(self, boxes1, boxes2):
        """
        Calculate 3D IoU between two sets of 3D bounding boxes
        Args:
            boxes1: [N, 7] (x, y, z, w, h, l, r)
            boxes2: [N, 7] (x, y, z, w, h, l, r)
        Returns:
            iou: [N] IoU values
        """
        # Extract box parameters
        x1, y1, z1, w1, h1, l1, r1 = boxes1.unbind(-1)
        x2, y2, z2, w2, h2, l2, r2 = boxes2.unbind(-1)
        
        # Calculate volumes
        vol1 = w1 * h1 * l1
        vol2 = w2 * h2 * l2
        
        # For simplicity, we approximate 3D IoU using axis-aligned bounding boxes
        # In practice, you might want to handle rotation properly
        
        # Calculate intersection bounds
        x_left = torch.max(x1 - w1/2, x2 - w2/2)
        x_right = torch.min(x1 + w1/2, x2 + w2/2)
        y_bottom = torch.max(y1 - h1/2, y2 - h2/2)
        y_top = torch.min(y1 + h1/2, y2 + h2/2)
        z_near = torch.max(z1 - l1/2, z2 - l2/2)
        z_far = torch.min(z1 + l1/2, z2 + l2/2)
        
        # Calculate intersection volume
        intersection_w = torch.clamp(x_right - x_left, min=0)
        intersection_h = torch.clamp(y_top - y_bottom, min=0)
        intersection_l = torch.clamp(z_far - z_near, min=0)
        
        intersection_vol = intersection_w * intersection_h * intersection_l
        
        # Calculate union volume
        union_vol = vol1 + vol2 - intersection_vol
        
        # Calculate IoU
        iou = intersection_vol / (union_vol + self.eps)
        
        return iou


class GIoU3DLoss(nn.Module):
    """Generalized 3D IoU Loss for better gradient flow"""
    def __init__(self, reduction='mean', eps=1e-7):
        super(GIoU3DLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, predict, target):
        """
        Args:
            predict: Predicted 3D bounding boxes [B, N, 7] (x, y, z, w, h, l, r)
            target: Ground truth 3D bounding boxes [B, N, 7] (x, y, z, w, h, l, r)
        """
        assert predict.shape == target.shape, f'predict & target shape do not match: {predict.shape} vs {target.shape}'
        
        # Handle invalid targets (marked with -1)
        valid_mask = (target != -1).all(dim=-1)  # [B, N]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predict.device, requires_grad=True)
        
        # Apply mask to get valid predictions and targets
        valid_predict = predict[valid_mask]  # [valid_boxes, 7]
        valid_target = target[valid_mask]    # [valid_boxes, 7]
        
        # Calculate GIoU
        giou = self.calculate_3d_giou(valid_predict, valid_target)
        
        # Convert GIoU to loss (1 - GIoU)
        giou_loss = 1.0 - giou
        
        if self.reduction == 'mean':
            return giou_loss.mean()
        elif self.reduction == 'sum':
            return giou_loss.sum()
        else:
            return giou_loss

    def calculate_3d_giou(self, boxes1, boxes2):
        """
        Calculate 3D GIoU between two sets of 3D bounding boxes
        """
        # Extract box parameters
        x1, y1, z1, w1, h1, l1, r1 = boxes1.unbind(-1)
        x2, y2, z2, w2, h2, l2, r2 = boxes2.unbind(-1)
        
        # Calculate volumes
        vol1 = w1 * h1 * l1
        vol2 = w2 * h2 * l2
        
        # Calculate intersection bounds
        x_left = torch.max(x1 - w1/2, x2 - w2/2)
        x_right = torch.min(x1 + w1/2, x2 + w2/2)
        y_bottom = torch.max(y1 - h1/2, y2 - h2/2)
        y_top = torch.min(y1 + h1/2, y2 + h2/2)
        z_near = torch.max(z1 - l1/2, z2 - l2/2)
        z_far = torch.min(z1 + l1/2, z2 + l2/2)
        
        # Calculate intersection volume
        intersection_w = torch.clamp(x_right - x_left, min=0)
        intersection_h = torch.clamp(y_top - y_bottom, min=0)
        intersection_l = torch.clamp(z_far - z_near, min=0)
        intersection_vol = intersection_w * intersection_h * intersection_l
        
        # Calculate union volume
        union_vol = vol1 + vol2 - intersection_vol
        
        # Calculate IoU
        iou = intersection_vol / (union_vol + self.eps)
        
        # Calculate enclosing box bounds
        x_left_c = torch.min(x1 - w1/2, x2 - w2/2)
        x_right_c = torch.max(x1 + w1/2, x2 + w2/2)
        y_bottom_c = torch.min(y1 - h1/2, y2 - h2/2)
        y_top_c = torch.max(y1 + h1/2, y2 + h2/2)
        z_near_c = torch.min(z1 - l1/2, z2 - l2/2)
        z_far_c = torch.max(z1 + l1/2, z2 + l2/2)
        
        # Calculate enclosing volume
        enclosing_w = x_right_c - x_left_c
        enclosing_h = y_top_c - y_bottom_c
        enclosing_l = z_far_c - z_near_c
        enclosing_vol = enclosing_w * enclosing_h * enclosing_l
        
        # Calculate GIoU
        giou = iou - (enclosing_vol - union_vol) / (enclosing_vol + self.eps)
        
        return giou


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for 3D bounding box regression"""
    def __init__(self, beta=1.0, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, predict, target):
        """
        Args:
            predict: Predicted 3D bounding boxes [B, N, 7] (x, y, z, w, h, l, r)
            target: Ground truth 3D bounding boxes [B, N, 7] (x, y, z, w, h, l, r)
        """
        assert predict.shape == target.shape, f'predict & target shape do not match: {predict.shape} vs {target.shape}'
        
        # Handle invalid targets (marked with -1)
        valid_mask = (target != -1).all(dim=-1)  # [B, N]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predict.device, requires_grad=True)
        
        # Apply mask to get valid predictions and targets
        valid_predict = predict[valid_mask]  # [valid_boxes, 7]
        valid_target = target[valid_mask]    # [valid_boxes, 7]
        
        # Calculate smooth L1 loss
        diff = torch.abs(valid_predict - valid_target)
        loss = torch.where(diff < self.beta, 
                          0.5 * diff.pow(2) / self.beta,
                          diff - 0.5 * self.beta)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ================================
# Combined Loss Classes
# ================================

class Combined3DBBoxLoss(nn.Module):
    """Combined loss for 3D bounding box regression"""
    def __init__(self, l1_weight=1.0, iou_weight=1.0, giou_weight=0.0):
        super(Combined3DBBoxLoss, self).__init__()
        self.l1_weight = l1_weight
        self.iou_weight = iou_weight
        self.giou_weight = giou_weight
        
        self.l1_loss = L1Loss()
        self.iou_loss = IoU3DLoss()
        if giou_weight > 0:
            self.giou_loss = GIoU3DLoss()

    def forward(self, predict, target):
        total_loss = 0.0
        
        if self.l1_weight > 0:
            l1_loss = self.l1_loss(predict, target)
            total_loss += self.l1_weight * l1_loss
        
        if self.iou_weight > 0:
            iou_loss = self.iou_loss(predict, target)
            total_loss += self.iou_weight * iou_loss
        
        if self.giou_weight > 0:
            giou_loss = self.giou_loss(predict, target)
            total_loss += self.giou_weight * giou_loss
        
        return total_loss