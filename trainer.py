from transformers import Trainer
import torch
from eval import evaluate_single
import numpy as np


class TracTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_predictions = []  # Store real predictions here

    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

            # Store real predictions for IoU computation
            if not prediction_loss_only:
                bbox_preds = outputs["bbox_3d_pred"]  # List of different shaped tensors
                bbox_labels = inputs["bbox_gts"]  # [B, D, H, W]
                bbox_masks = inputs["bbox_masks"]  # [B, D, H, W]

                # Store the real data
                batch_data = {
                    "bbox_preds": bbox_preds,
                    "bbox_labels": bbox_labels,
                    "bbox_masks": bbox_masks,
                }
                self.eval_predictions.append(batch_data)

                # Return dummy tensors to satisfy HuggingFace (these will be ignored)
                dummy_pred = torch.zeros(len(bbox_preds), 1)  # [B, 1]
                dummy_label = torch.zeros(bbox_labels.shape[0], 1)  # [B, 1]

                return (loss, dummy_pred, dummy_label)

            return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to compute IoU from stored predictions"""
        self.eval_predictions = []

        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        iou_score = self.compute_iou_from_stored_predictions()

        eval_results[f"{metric_key_prefix}_iou"] = iou_score

        return eval_results

    def compute_iou_from_stored_predictions(self):
        """Compute IoU from stored variable-shape predictions"""
        total_iou = 0
        total_samples = 0

        ious = []
        for batch_data in self.eval_predictions:
            bbox_preds = batch_data["bbox_preds"]  # List of variable shapes
            bbox_labels = batch_data["bbox_labels"]  # [B, D, H, W]
            bbox_masks = batch_data["bbox_masks"]  # [B, D, H, W]

            for pred, label in zip(bbox_preds, bbox_labels):
                pred, gt, iou = evaluate_single(pred, label)
                if len(iou) > 0:
                    print("sample iou compute metric", iou)
                    ious.extend(iou)

        return np.mean(ious)
