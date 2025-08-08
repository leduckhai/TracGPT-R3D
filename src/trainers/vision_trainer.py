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
        self.computer_loss=kwargs.get("computer_loss")
    

   
 
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
