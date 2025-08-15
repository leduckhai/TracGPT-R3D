from transformers import Trainer
import torch
from torchvision.ops import box_iou
from tqdm import tqdm
from collections import defaultdict

class Bbox2DTrainer(Trainer):
    def __init__(self, tracker, *args, **kwargs):
        self.tracker=tracker
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        images = [img.to(model.device) for img in inputs["images"]]  # list of (D, H, W) tensors
        bboxes = inputs["bboxes"]

        batch_input_images = []
        targets = []

        for vol, vol_bboxes in zip(images, bboxes):
            # vol: (D, H, W), vol_bboxes: list of bboxes per slice
            for slice_img, slice_bboxes in zip(vol, vol_bboxes):
                # slice_img: (H, W)
                rgb_slice = slice_img.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
                batch_input_images.append(rgb_slice)

                # Make target dict for this slice
                target_dict = {
                    "boxes": torch.tensor(slice_bboxes, dtype=torch.float32, device=model.device),
                    "labels": torch.ones(len(slice_bboxes), dtype=torch.int64, device=model.device)
                }
                targets.append(target_dict)

        losses = model(batch_input_images, targets)
        total_loss = sum(losses.values())/len(targets)
        if self.state.global_step % self.args.logging_steps == 0:
            log_dict={
                "train/loss":total_loss.item(),
            }
            self.tracker.log(log_dict, step=self.state.global_step)
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss
    
    @torch.no_grad()
    def evaluate(self,  iou_threshold=0.5, conf_threshold=0.5,ignore_keys=None):
        log_period=5
        model = self.model
        eval_dataloader = self.get_eval_dataloader()
        model.eval()
        total_images = 0
        total_true_positives = 0
        total_predictions = 0
        total_ground_truths = 0

        eval_loop = tqdm(enumerate(eval_dataloader), 
                    total=len(eval_dataloader),
                    desc="Evaluating",
                    leave=True)
    
        for i, sample in eval_loop:
            volumes = sample["images"]
            targets_batch = sample["bboxes"]
            # volumes: list of (D, H, W)
            batch_input_images = []
            batch_targets = []

            for vol, vol_targets in zip(volumes, targets_batch):
                # vol: (D, H, W), vol_targets: list of bboxes per slice
                for slice_img, slice_bboxes in zip(vol, vol_targets):
                    # convert grayscale slice to RGB
                    rgb_slice = slice_img.unsqueeze(0).repeat(3, 1, 1).to(model.device)
                    batch_input_images.append(rgb_slice)

                    # slice target dict
                    target_dict = {
                        "boxes": torch.tensor(slice_bboxes, dtype=torch.float32, device=model.device),
                        "labels": torch.ones(len(slice_bboxes), dtype=torch.int64, device=model.device)
                    }
                    batch_targets.append(target_dict)

            # Run model
            outputs = model(batch_input_images)

            # Evaluate per slice
            for pred, gt in zip(outputs, batch_targets):
                pred_boxes = pred["boxes"]
                pred_scores = pred["scores"]
                gt_boxes = gt["boxes"]

                # filter by confidence
                conf_mask = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[conf_mask]

                total_predictions += len(pred_boxes)
                total_ground_truths += len(gt_boxes)

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    matches = (ious >= iou_threshold).sum(dim=1) > 0
                    total_true_positives += matches.sum().item()

            total_images += len(batch_input_images)
            if i % log_period == 0:
                current_precision = total_true_positives / total_predictions if total_predictions > 0 else 0
                current_recall = total_true_positives / total_ground_truths if total_ground_truths > 0 else 0
                eval_loop.set_postfix({
                    "Precision": current_precision,
                    "Recall": current_recall,
                    "TP": total_true_positives,
                    "Predictions": total_predictions,
                    "Ground Truths": total_ground_truths,
                    "Images": total_images
                })
                self.tracker.log({
                    "eval/precision": current_precision,
                    "eval/recall": current_recall,
                    "eval/true_positives": total_true_positives,
                    "eval/predictions": total_predictions,
                    "eval/ground_truths": total_ground_truths,
                    "eval/images": total_images
                }, step=self.state.global_step)
        precision = total_true_positives / total_predictions if total_predictions > 0 else 0
        recall = total_true_positives / total_ground_truths if total_ground_truths > 0 else 0

        print(f"Evaluation results - Precision: {precision:.4f}, Recall: {recall:.4f} over {total_images} slices")
        return {
            "eval_precision": precision,
            "eval_recall": recall,
            "total_images": total_images
        }