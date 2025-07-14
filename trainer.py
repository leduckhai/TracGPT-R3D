from transformers import Trainer
import torch
from eval import evaluate_single
import numpy as np
import wandb
from transformers.cache_utils import DynamicCache

class TracTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_predictions = []
        self.current_step = 0
    def log(self, logs, start_time=None):
        if hasattr(self, '_current_outputs') and self._current_outputs:
            for key, value in self._current_outputs.items():
                if key.endswith('_loss') and key != 'loss' and torch.is_tensor(value):
                    logs[key] = value.item()
        
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        outputs = model(**inputs)
        
        # Recursively check all tensors in outputs
        def _check_nan(obj, path=""):
            if isinstance(obj, torch.Tensor):
                if torch.is_floating_point(obj) and torch.isnan(obj).any():
                    print(f"NaN detected in {path} | Shape: {obj.shape} | Dtype: {obj.dtype}")
                    return True
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    if _check_nan(item, f"{path}[{i}]"):
                        return True
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if _check_nan(value, f"{path}.{key}"):
                        return True
            elif isinstance(obj, DynamicCache):  # Special handling for cache
                for layer_idx in range(len(obj.key_cache)):
                    if _check_nan(obj.key_cache[layer_idx], f"{path}.key_cache[{layer_idx}]"):
                        return True
                    if _check_nan(obj.value_cache[layer_idx], f"{path}.value_cache[{layer_idx}]"):
                        return True
            return False
        
        if _check_nan(outputs, "outputs"):
            print("\n=== NaN DETECTED ===")
            self._log_debug_info(model, inputs)
            breakpoint()
            
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _log_debug_info(self, model, inputs):
        """Log detailed debugging information"""
        print("\n=== Model State ===")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Input shapes: { {k: v.shape for k,v in inputs.items() if torch.is_tensor(v)} }")
        
        print("\n=== Input Data Check ===")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                print(f"{k}:")
                print(f"- NaN: {torch.isnan(v).any().item()}")
                print(f"- Inf: {torch.isinf(v).any().item()}")
                print(f"- Min: {v.min().item()}")
                print(f"- Max: {v.max().item()}")
        
        if hasattr(model, 'peft_config'):
            print("\n=== LoRA Weights Check ===")
            for name, param in model.named_parameters():
                if 'lora' in name.lower():
                    print(f"{name}:")
                    print(f"- Weight NaN: {torch.isnan(param).any().item()}")
                    print(f"- Grad NaN: {param.grad is not None and torch.isnan(param.grad).any().item()}")
    def training_step(self, model, inputs, num_items_in_batch=None):
       
        with torch.no_grad():
            outputs = model(**inputs)
        self._current_outputs = outputs
        loss = super().training_step(model, inputs, num_items_in_batch)        
        if isinstance(outputs, dict) and "bbox_3d_loss" in outputs and wandb.run is not None:
            wandb.log({
                "train/bbox_3d_loss": outputs["bbox_3d_loss"].item(),
                "step": self.state.global_step
            })
        
        return loss
    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

            # Store real predictions for IoU computation
            # if not prediction_loss_only:
            bbox_preds = outputs["bbox_3d_pred"]
            bbox_loss = outputs["bbox_3d_loss"]
            bbox_labels = inputs["bbox_gts"]
            bbox_masks = inputs["bbox_masks"]

            # Store the real data
            batch_data = {
                "bbox_preds": bbox_preds,
                "bbox_labels": bbox_labels,
                "bbox_masks": bbox_masks,
                "bbox_3d_loss": bbox_loss,
            }
            self.eval_predictions.append(batch_data)

            # Return dummy tensors to satisfy HuggingFace
            dummy_pred = torch.zeros(len(bbox_preds), 1)
            dummy_label = torch.zeros(bbox_labels.shape[0], 1)

            return (loss, dummy_pred, dummy_label)

        # return (loss, None, None)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Override evaluation_loop to ensure our custom logic runs"""
        print(f"Starting custom evaluation loop with prefix: {metric_key_prefix}")

        # Clear previous predictions
        self.eval_predictions = []

        # Call parent evaluation_loop
        eval_loop_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        print(f"Collected {len(self.eval_predictions)} prediction batches")

        iou_score = self.compute_iou_from_stored_predictions()
        bbox_losses = [batch["bbox_3d_loss"].item() for batch in self.eval_predictions]
        mean_bbox_loss = np.mean(bbox_losses) if bbox_losses else float("nan")

        # Log to wandb
        if wandb.run is not None:
            wandb.log(
                {
                    f"{metric_key_prefix}/bbox_3d_loss": mean_bbox_loss,
                    f"{metric_key_prefix}/iou": (
                        iou_score if iou_score is not None else float("nan")
                    ),
                    "step": self.current_step,
                }
            )

        eval_loop_output.metrics.update(
            {
                f"{metric_key_prefix}_bbox_3d_loss": mean_bbox_loss,
                f"{metric_key_prefix}_iou": (
                    iou_score if iou_score is not None else float("nan")
                ),
            }
        )

        self.current_step += 1
        return eval_loop_output

    def compute_iou_from_stored_predictions(self):
        """Compute IoU from stored variable-shape predictions"""
        if len(self.eval_predictions) == 0:
            print("No predictions stored for IoU computation")
            return None

        ious = []

        for batch_idx, batch_data in enumerate(self.eval_predictions):
            bbox_preds = batch_data["bbox_preds"]
            bbox_labels = batch_data["bbox_labels"]
            bbox_masks = batch_data["bbox_masks"]

            for pred, label, mask in zip(bbox_preds, bbox_labels, bbox_masks):
                
                    pred, gt, iou = evaluate_single(pred, label, mask)
                    if len(iou) > 0:
                        print(f"Batch {batch_idx}: IoU = {iou}")
                        ious.extend(iou)
              

        if len(ious) == 0:
            print("No valid IoU values computed")
            return None

        mean_iou = np.mean(ious)
        print(f"Mean IoU: {mean_iou} (from {len(ious)} values)")
        return float(mean_iou)
