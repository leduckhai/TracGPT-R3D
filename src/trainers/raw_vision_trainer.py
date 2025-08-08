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
        # self.tracker=WandbTracker()

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
        print("Evaluating")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
        model = self.model.eval()
        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
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
                bbox_GCA_metric=self.eval_metric_tracker.calculate_metrics_bbox_criteria("GCA",bbox_GCA_pred,bbox_GCA)
                bbox_Koedam_metric=self.eval_metric_tracker.calculate_metrics_bbox_criteria("Koedam",bbox_Koedam_pred,bbox_Koedam)
                bbox_MTA_metric=self.eval_metric_tracker.calculate_metrics_bbox_criteria("MTA",bbox_MTA_pred,bbox_MTA)
                bbox_status_metric=self.eval_metric_tracker.calculate_metrics_bbox_criteria("Status",status_pred,status_criteria)
            
                if self.state.global_step % self.args.logging_steps == 0:
                    log_dict={
                        
                        "val/bbox_GCA_loss":metric_loss["GCA"].item(),
                        "val/bbox_Koedam_loss":metric_loss["Koedam"].item(),
                        "val/bbox_MTA_loss":metric_loss["MTA"].item(),
                        "val/status_loss":metric_loss["status"].item(),
                        "val/bbox_GCA_metric":bbox_GCA_metric,
                        "val/bbox_Koedam_metric":bbox_Koedam_metric,
                        "val/bbox_MTA_metric":bbox_MTA_metric,
                        "val/status_metric":bbox_status_metric
                    }
                    self.tracker.log(log_dict, self.state.global_step)
        for metric_name, metric_group_val in self.eval_metric_tracker.get_global_bbox_metrics().items():
            print("Metric val",metric_name)
            for metric_key,metric_val in metric_group_val.items():
                print(metric_key,np.mean(metric_val)) 
                self.tracker.log({f"val/mean_{metric_name}_{metric_key}":np.mean(metric_val)},self.state.global_step)
        return 

