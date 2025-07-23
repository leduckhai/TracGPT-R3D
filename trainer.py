from transformers import Trainer
import torch
from eval import evaluate_single
import numpy as np
import wandb
from transformers.cache_utils import DynamicCache
from collections import defaultdict

class TracTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_predictions = defaultdict(list)
        self.current_step = 0
        self.gradient_log_freq = 10
    def log(self, logs, start_time=None):
        if hasattr(self, '_current_outputs') and self._current_outputs:
            for key, value in self._current_outputs.items():
                if key.endswith('_loss') and key != 'loss' and torch.is_tensor(value):
                    logs[key] = value.item()
        
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
    
    def log_gradient_histograms(self, model, step):
        """Log gradient histograms to wandb"""
        if wandb.run is None:
            return
            
        for name, param in model.named_parameters():
            if param.grad is not None:
                wandb.log({
                    f"grad_hist/{name}": wandb.Histogram(param.grad.cpu().numpy())
                }, step=step)
    
    def log_gradient_norms(self, model, step):
        """Log gradient norms and statistics"""
        if wandb.run is None:
            return
            
        total_norm = 0.0
        param_count = 0
        
        gradient_logs = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log individual layer gradient norms
                gradient_logs[f"grad_norm/{name}"] = param_norm.item()
                gradient_logs[f"grad_mean/{name}"] = param.grad.mean().item()
                gradient_logs[f"grad_std/{name}"] = param.grad.std().item()
                gradient_logs[f"grad_max/{name}"] = param.grad.max().item()
                gradient_logs[f"grad_min/{name}"] = param.grad.min().item()
        
        # Log total gradient norm
        total_norm = total_norm ** 0.5
        gradient_logs["grad_norm/total"] = total_norm
        gradient_logs["grad_norm/average"] = total_norm / max(param_count, 1)
        
        wandb.log(gradient_logs, step=step)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        (images,
        input_ids,
        attention_mask,
        positive_centers,
        labels,
        center_bbox_gts,
        bbox_mask,
        position_ids,
        answer_types,
        questions,
        answers,
        corner_bbox_gts) = inputs.values()
        
        # Prepare filtered inputs for the parent training_step
        filter_inputs = {
            'input_ids': input_ids,
            'images': images,
            'bbox_gts': center_bbox_gts,
            'bbox_masks': bbox_mask,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        
        loss = super().training_step(model, filter_inputs, num_items_in_batch)
        
        if self.state.global_step % self.gradient_log_freq == 0:
            self.log_gradient_norms(model, self.state.global_step)
            
            self.log_gradient_histograms(model, self.state.global_step)
        
        with torch.no_grad(): 
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox_gts=center_bbox_gts,
                bbox_masks=bbox_mask,
                labels=labels,
                position_ids=position_ids
            )
            
            if isinstance(outputs, dict) and wandb.run is not None and "aux_loss" in outputs:
                aux_metrics = outputs["aux_loss"]
                for metric_key, metric_value in aux_metrics.items():
                    wandb.log({
                        f"train/{metric_key}": metric_value,
                        "step": self.state.global_step
                    })
        
        return loss
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

        # Initialize metrics storage
        self.eval_metrics = {
            'bbox_loss': [],
            'aux_metrics': defaultdict(list)
        }

        # Run parent evaluation
        eval_loop_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Calculate and log metrics
        metrics = {}
        if self.eval_metrics['bbox_loss']:
            metrics[f'{metric_key_prefix}/bbox_loss'] = np.mean(self.eval_metrics['bbox_loss'])

        # Process auxiliary metrics
        for metric_key, values in self.eval_metrics['aux_metrics'].items():
            metrics[f'{metric_key_prefix}/{metric_key}'] = np.mean(values)

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                **metrics,
                "step": self.state.global_step
            })

        eval_loop_output.metrics.update({
            k.replace('/', '_'): v for k, v in metrics.items()
        })

        self.state.global_step += 1
        return eval_loop_output
    
