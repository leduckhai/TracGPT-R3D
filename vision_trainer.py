from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from tqdm import tqdm
import wandb  # optional, remove if not using wandb

class TracVisionTrainer(Trainer):
    def __init__(self, *args, eval_metrics=None, log_gradients=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_predictions = defaultdict(list)
        self.current_step = 0
        self.gradient_log_freq = 10
        self.log_gradients = log_gradients
        self.eval_metrics = eval_metrics or ['bbox_loss', 'pos_loss', 'size_loss', 'conf_loss']
        
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        """
        Compute the custom loss for TracVisionModel
        """
        # print("inputs",inputs)
        images = inputs.get("images")
        bbox_gts = inputs.get("center_bbox_gts")
        bbox_masks = inputs.get("bbox_masks")
        labels = inputs.get("labels")
        if bbox_gts is None:
            print("bbox_gts is None")
            return 
        if bbox_masks is None:
            print("bbox_masks is None")
            return
        outputs = model(
            images=images,
            bbox_gts=bbox_gts,
            bbox_masks=bbox_masks,
            labels=labels,
            return_dict=True
        )
        # print("outputs",outputs)
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=images.device, requires_grad=True)
        # print("loss",loss)
        if outputs.aux_loss is  None:
            print("outputs.aux_loss is None")
            self. _current_outputs = {
               'loss': loss,
           }
        else:
            self._current_outputs = {
                'loss': loss,
                **outputs.aux_loss
            }
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Custom training step with gradient logging
        """
        loss = super().training_step(model, inputs)
        
        if self.log_gradients and self.state.global_step % self.gradient_log_freq == 0:
            self._log_gradients(model)
            
        return loss

    def _log_gradients(self, model):
        """
        Log gradient statistics to wandb
        """
        if wandb.run is None:
            return
            
        gradient_logs = {}
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                # Gradient norms
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log statistics
                prefix = f"gradients/{name.replace('.', '/')}"
                gradient_logs.update({
                    f"{prefix}/norm": param_norm.item(),
                    f"{prefix}/mean": param.grad.mean().item(),
                    f"{prefix}/std": param.grad.std().item(),
                    f"{prefix}/max": param.grad.max().item(),
                    f"{prefix}/min": param.grad.min().item(),
                })
                
                # Histograms
                wandb.log({
                    f"grad_hist/{name}": wandb.Histogram(param.grad.cpu().numpy())
                }, step=self.state.global_step)
        
        # Log total gradient statistics
        if param_count > 0:
            total_norm = total_norm ** 0.5
            gradient_logs.update({
                "gradients/total_norm": total_norm,
                "gradients/avg_norm": total_norm / param_count
            })
            
            wandb.log(gradient_logs, step=self.state.global_step)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Custom evaluation with bbox-specific metrics
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Initialize metrics
        metrics = {key: [] for key in self.eval_metrics}
        iou_scores = []
        
        model = self.model.eval()
        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = model(
                    images=inputs["images"],
                    bbox_gts=inputs["bbox_gts"],
                    bbox_masks=inputs["bbox_masks"],
                    return_dict=True
                )
            
            # Collect metrics
            for key in self.eval_metrics:
                if key in outputs.aux_loss:
                    metrics[key].append(outputs.aux_loss[key])
            
            # Calculate IoU if predictions available
            if outputs.predicts is not None:
                batch_ious = self._calculate_batch_iou(
                    outputs.predicts, 
                    inputs["bbox_gts"],
                    inputs["bbox_masks"]
                )
                iou_scores.extend(batch_ious)
        
        # Compute average metrics
        avg_metrics = {
            f"{metric_key_prefix}_{key}": np.mean(values) 
            for key, values in metrics.items() if values
        }
        
        if iou_scores:
            avg_metrics[f"{metric_key_prefix}_iou"] = np.mean(iou_scores)
        
        if wandb.run is not None:
            wandb.log({
                **avg_metrics,
                "step": self.state.global_step
            })
        
        return avg_metrics

    # def _calculate_batch_iou(self, pred_boxes, gt_boxes, masks):
    #     """
    #     Calculate IoU for a batch of predictions
    #     """
    #     ious = []
    #     for pred, gt, mask in zip(pred_boxes, gt_boxes, masks):
    #         if mask.any():  # Only calculate for valid boxes
    #             # Convert to numpy for calculation
    #             pred = pred.detach().cpu().numpy()
    #             gt = gt.detach().cpu().numpy()
    #             mask = mask.detach().cpu().numpy()
                
    #             # Calculate IoU for each valid box
    #             for p, g, m in zip(pred, gt, mask):
    #                 if m:  # Only for valid masks
    #                     iou = self._calculate_iou(p, g)
    #                     ious.append(iou)
    #     return ious

    # def _calculate_iou(self, box1, box2):
    #     """
    #     Simple 3D IoU calculation (simplified example)
    #     """
    #     # This is a placeholder - implement your actual 3D IoU calculation
    #     # For demonstration, we'll use center distance as a proxy
    #     center_dist = np.linalg.norm(box1[:3] - box2[:3])
    #     return 1.0 / (1.0 + center_dist)  # Inverse distance as proxy for IoU

    def log(self, logs, start_time=None, **kwargs):
        """
        Enhanced logging with auxiliary losses
        """
        # Add auxiliary losses from current outputs
        if hasattr(self, '_current_outputs'):
            for key, value in self._current_outputs.items():
                if key.endswith('_loss') and torch.is_tensor(value):
                    logs[key] = value.item()
        
        super().log(logs, start_time=start_time, **kwargs)