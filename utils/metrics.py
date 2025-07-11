import numpy as np

def dice_score(preds, labels, smooth=1e-5):
    """
    Compute Dice coefficient for binary 3D masks.
    Assumes inputs are numpy arrays of shape [B, D, H, W]
    """
    dice_scores = []
    for pred, label in zip(preds, labels):
        pred = pred > 0.5  # Threshold if needed
        label = label > 0.5
        intersection = np.logical_and(pred, label).sum()
        union = pred.sum() + label.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    return np.mean(dice_scores)