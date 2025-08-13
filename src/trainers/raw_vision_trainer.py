from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from tqdm import tqdm
import os
import wandb 
from src.trainers.metrics import MetricTracker
import torch.nn as nn
from src.trainers.tracker import WandbTracker
class RawVisionTrainer(Trainer):
    def __init__(self, *args, eval_metrics=None, log_gradients=False,tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_predictions = defaultdict(list)
        self.current_step = 0
        self.gradient_log_freq = 15
        self.log_gradients = log_gradients
        self.eval_metrics = eval_metrics or ['bbox_loss', 'pos_loss', 'size_loss', 'conf_loss']
        self.gradient_path = "gradient/gradient_logs.txt"
        self.metric_tracker=MetricTracker()
        self.eval_metric_tracker=MetricTracker()
        self.loss= nn.CrossEntropyLoss()
        self.tracker=tracker

    def compute_model_loss(self, bbox_GCA, bbox_Koedam, bbox_MTA,status_criteria, outputs ):
        subtask_out,finaltask_out=outputs
        metric_loss={}
        metric_loss["GCA"] = self.loss(subtask_out[:,0,:],bbox_GCA.long().to("cuda"))
        metric_loss["Koedam"] = self.loss(subtask_out[:,1,:],bbox_Koedam.long().to("cuda"))
        metric_loss["MTA"] = self.loss(subtask_out[:,2,:],bbox_MTA.long().to("cuda"))
        metric_loss["status"] = self.loss(finaltask_out.squeeze(1), status_criteria.long().to("cuda"))
        loss = sum(metric_loss.values())
        return loss,metric_loss
 
    def training_step(self, model, inputs, num_items_in_batch=None):
        images=inputs.get("images").to(model.device)
        bbox_GCA=inputs.get("bbox_GCA").to(model.device)
        bbox_Koedam=inputs.get("bbox_Koedam").to(model.device)
        bbox_MTA=inputs.get("bbox_MTA").to(model.device)
        status_criteria=inputs.get("status_criteria")
        outputs=model(images )
        subtask_out,finaltask_out=outputs
        bbox_GCA_pred=subtask_out[:,0,:].squeeze(1)
        bbox_Koedam_pred=subtask_out[:,1,:].squeeze(1)
        bbox_MTA_pred=subtask_out[:,2,:].squeeze(1)
        status_pred=finaltask_out.squeeze(1)
        loss,metric_loss = self.compute_model_loss( bbox_GCA,bbox_Koedam,bbox_MTA,status_criteria,outputs)     
        bbox_GCA_metric=self.metric_tracker.calculate_metrics_bbox_criteria("GCA",bbox_GCA_pred,bbox_GCA)
        bbox_Koedam_metric=self.metric_tracker.calculate_metrics_bbox_criteria("Koedam",bbox_Koedam_pred,bbox_Koedam)
        bbox_MTA_metric=self.metric_tracker.calculate_metrics_bbox_criteria("MTA",bbox_MTA_pred,bbox_MTA)
        bbox_status_metric=self.metric_tracker.calculate_metrics_bbox_criteria("Status",status_pred,status_criteria)
      
        if self.state.global_step % self.args.logging_steps == 0:
            log_dict={
                
                "train/bbox_GCA_loss":metric_loss["GCA"].item(),
                "train/bbox_Koedam_loss":metric_loss["Koedam"].item(),
                "train/bbox_MTA_loss":metric_loss["MTA"].item(),
                "train/status_loss":metric_loss["status"].item(),
                "train/bbox_GCA_metric":bbox_GCA_metric,
                "train/bbox_Koedam_metric":bbox_Koedam_metric,
                "train/bbox_MTA_metric":bbox_MTA_metric,
                "train/status_metric":bbox_status_metric
            }
            self.tracker.log(log_dict, self.state.global_step)
        loss.backward()     
        # if self.state.global_step % self.gradient_log_freq == 0:
        #     self._log_gradient_stats(model, self.state.global_step)
        # gradient_anomaly = self._check_gradients()
        # self._clip_gradients()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
                
        return loss.detach()
 
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        print(f"Evaluating with metric key prefix -- {metric_key_prefix}")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        total_loss = 0.0
        num_batches = 0

        model = self.model.eval()
        mean_loss_GCA=[]
        mean_loss_Koedam=[]
        mean_loss_MTA=[]
        mean_loss_status=[]
        mean_metric_GCA=defaultdict(list)
        mean_metric_Koedam=defaultdict(list)
        mean_metric_MTA=defaultdict(list)
        mean_metric_status=defaultdict(list)
        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                images = inputs.get("images").to(model.device)
                bbox_GCA = inputs.get("bbox_GCA").to(model.device)
                bbox_Koedam = inputs.get("bbox_Koedam").to(model.device)
                bbox_MTA = inputs.get("bbox_MTA").to(model.device)
                status_criteria = inputs.get("status_criteria")
                
                outputs = model(images)
                subtask_out, finaltask_out = outputs
                loss, metric_loss = self.compute_model_loss(bbox_GCA, bbox_Koedam, bbox_MTA, status_criteria, outputs)
                
                mean_loss_GCA.append(metric_loss["GCA"].item())
                mean_loss_Koedam.append(metric_loss["Koedam"].item())
                mean_loss_MTA.append(metric_loss["MTA"].item())
                mean_loss_status.append(metric_loss["status"].item())
                
                total_loss += loss.item()
                num_batches += 1
                for k,v in metric_loss.items():
                    mean_metric_GCA[k].append(v.item())
                    mean_metric_Koedam[k].append(v.item())
                    mean_metric_MTA[k].append(v.item())
                    mean_metric_status[k].append(v.item())


                bbox_GCA_metric = self.eval_metric_tracker.calculate_metrics_bbox_criteria("GCA", subtask_out[:,0,:].squeeze(1), bbox_GCA)
                bbox_Koedam_metric = self.eval_metric_tracker.calculate_metrics_bbox_criteria("Koedam", subtask_out[:,1,:].squeeze(1), bbox_Koedam)
                bbox_MTA_metric = self.eval_metric_tracker.calculate_metrics_bbox_criteria("MTA", subtask_out[:,2,:].squeeze(1), bbox_MTA)
                bbox_status_metric = self.eval_metric_tracker.calculate_metrics_bbox_criteria("Status", finaltask_out.squeeze(1), status_criteria)

                if self.state.global_step % self.args.logging_steps == 0:
                    log_dict = {
                        f"{metric_key_prefix}/loss":loss.item(),
                        f"{metric_key_prefix}/bbox_GCA_loss": metric_loss["GCA"].item(),
                        f"{metric_key_prefix}/bbox_Koedam_loss": metric_loss["Koedam"].item(),
                        f"{metric_key_prefix}/bbox_MTA_loss": metric_loss["MTA"].item(),
                        f"{metric_key_prefix}/status_loss": metric_loss["status"].item(),
                        f"{metric_key_prefix}/bbox_GCA_metric": bbox_GCA_metric,
                        f"{metric_key_prefix}/bbox_Koedam_metric": bbox_Koedam_metric,
                        f"{metric_key_prefix}/bbox_MTA_metric": bbox_MTA_metric,
                        f"{metric_key_prefix}/status_metric": bbox_status_metric
                    }
                    self.tracker.log(log_dict, self.state.global_step)
        mean_loss_GCA = sum(mean_loss_GCA) / len(mean_loss_GCA)
        mean_loss_Koedam = sum(mean_loss_Koedam) / len(mean_loss_Koedam)
        mean_loss_MTA = sum(mean_loss_MTA) / len(mean_loss_MTA)
        mean_loss_status = sum(mean_loss_status) / len(mean_loss_status)
        mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
        mean_metric_GCA = {k: sum(v) / len(v) for k, v in mean_metric_GCA.items()}
        mean_metric_Koedam = {k: sum(v) / len(v) for k, v in mean_metric_Koedam.items()}
        mean_metric_MTA = {k: sum(v) / len(v) for k, v in mean_metric_MTA.items()}
        mean_metric_status = {k: sum(v) / len(v) for k, v in mean_metric_status.items()}
        self.tracker.log({
            f"{metric_key_prefix}/mean_loss": mean_loss,
            f"{metric_key_prefix}/mean_bbox_GCA_loss": mean_loss_GCA,
            f"{metric_key_prefix}/mean_bbox_Koedam_loss": mean_loss_Koedam,
            f"{metric_key_prefix}/mean_bbox_MTA_loss": mean_loss_MTA,
            f"{metric_key_prefix}/mean_status_loss": mean_loss_status,
        }, self.state.global_step)
        return {
            f"{metric_key_prefix}_loss": mean_loss,  
            **{f"{metric_key_prefix}_{k}": v for k, v in self.eval_metric_tracker.get_global_bbox_metrics().items()}  # Optional: Custom metrics
        }
