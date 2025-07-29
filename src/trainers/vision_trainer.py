from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from tqdm import tqdm
import os
import wandb 
class TracVisionTrainer(Trainer):
    def __init__(self, *args, eval_metrics=None, log_gradients=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_predictions = defaultdict(list)
        self.current_step = 0
        self.gradient_log_freq = 15
        self.log_gradients = log_gradients
        self.eval_metrics = eval_metrics or ['bbox_loss', 'pos_loss', 'size_loss', 'conf_loss']
        self.gradient_path = "gradient/gradient_logs.txt"
        self._init_gradient_logging()  
    def _init_gradient_logging(self):
        """Initialize gradient log file (overwrites existing)"""
        try:
            os.makedirs(os.path.dirname(self.gradient_path), exist_ok=True)
            
            with open(self.gradient_path, 'w') as f:
                headers = [
                    f"{'Step':<8}",
                    f"{'Layer':<40}",
                    f"{'Mean':>12}",
                    f"{'Max':>12}",
                    f"{'Std':>12}",
                    f"{'Param/Mean':>12}",
                    f"{'NaN%':>6}"
                ]
                f.write(" ".join(headers) + "\n")
                f.write("-" * 100 + "\n")
                
            self.gradient_buffer = []
            self.buffer_size = 20
            
        except Exception as e:
            print(f"⚠️ Failed to initialize gradient logging: {str(e)}")
            self.log_gradients = False
    def _check_gradients(self):
        """Detect NaN/exploding gradients before optimizer step"""
        nan_found = False
        explode_found = False
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"⚠️ NaN gradients in {name}!")
                    nan_found = True
                if (param.grad.abs() > 1e3).any():
                    print(f"🚨 Exploding gradients in {name} (max={param.grad.abs().max():.1e})")
                    explode_found = True
                    
        return nan_found or explode_found

    def _clip_gradients(self):
        """Layer-wise gradient clipping with logging"""
        global_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if "bbox_head" in name:
                    global_norms[name] = 1.0  # Tighter clipping
                elif "vision_tower" in name:
                    global_norms[name] = 5.0  # Looser clipping
                else:
                    global_norms[name] = 2.0  # Default

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    [param], 
                    max_norm=global_norms[name],
                    norm_type=2
                )

    def _log_gradient_stats(self, model, step):
        print("Logging gradient stats...")
        print(f"Writing to: {os.path.abspath(self.gradient_path)}")  # Check full path
        print(f"File exists: {os.path.exists(self.gradient_path)}")  # False = path issue
        try:
            # Open file in append mode
            with open(self.gradient_path, 'a') as f:
                f.write(f"-------Global step: {self.state.global_step:<8}\n")
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.detach()
                        param_data = param.detach()
                        
                        nan_count = torch.isnan(grad).sum().item()
                        total_elements = grad.numel()
                        
                        if nan_count == total_elements:
                            stats = f"{name:<40} {'NaN':>12} {'NaN':>12} {'NaN':>12} {'NaN':>12} {100.0:>6.1f}%"
                        else:
                            grad_mean = grad[~torch.isnan(grad)].mean().item()
                            grad_max = grad[~torch.isnan(grad)].max().item()
                            grad_std = grad[~torch.isnan(grad)].std().item()
                            param_ratio = abs(param_data.mean().item() / (grad_mean + 1e-9))
                            nan_pct = (nan_count / total_elements) * 100
                            
                            stats = (
                                f"{name:<40} "
                                f"{grad_mean:>12.4e} "
                                f"{grad_max:>12.4e} "
                                f"{grad_std:>12.4e} "
                                f"{param_ratio:>12.2f} "
                                f"{nan_pct:>6.1f}%"
                            )
                        
                        f.write(f"{stats}\n")

                # Flush immediately to ensure writes
                f.flush()

        except Exception as e:
            print(f"⚠️ Failed to log gradients at step {step}: {str(e)}")
            if torch.cuda.is_available():
                print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the custom loss for TracVisionModel"""
        images = inputs.get("images")
        bbox_gts = inputs.get("center_bbox_gts")
        bbox_masks = inputs.get("bbox_masks")
        labels = inputs.get("labels")
        
        if bbox_gts is None:
            print("⚠️ bbox_gts is None")
            return 
        if bbox_masks is None:
            print("⚠️ bbox_masks is None")
            return
            
        outputs = model(
            images=images,
            bbox_gts=bbox_gts,
            bbox_masks=bbox_masks,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=images.device, requires_grad=True)
        
        self._current_outputs = {
            'loss': loss,
            **(outputs.aux_loss if outputs.aux_loss is not None else {})
        }
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
    
        loss = self.compute_loss(model, inputs)        
        loss.backward()        
        if self.state.global_step % self.gradient_log_freq == 0:
            self._log_gradient_stats(model, self.state.global_step)
          
        gradient_anomaly = self._check_gradients()
        
        self._clip_gradients()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
                
        return loss.detach()
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

def print_gradient_stats_to_file(model, step, file_path, print_console=False):
    """Log gradient statistics to a file with proper formatting.
    
    Args:
        model: PyTorch model
        step: Training step/epoch
        file_path: Output file path
        print_console: Whether to also print to console
    """
    header = f"\n{'Step':<8} {'Layer':<30} {'Mean':>10} {'Max':>10} {'Std':>10}\n"
    header += "-" * 68
    
    lines = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            line = (f"{step:<8} {name:<30} "
                   f"{param.grad.mean().item():>10.4f} "
                   f"{param.grad.max().item():>10.4f} "
                   f"{param.grad.std().item():>10.4f}")
            lines.append(line)
    
    with open(file_path, 'a') as f:
        if step == 0:
            f.write(header + "\n")
        
        f.write("\n".join(lines) + "\n")
    
    if print_console:
        print(header)
        print("\n".join(lines))