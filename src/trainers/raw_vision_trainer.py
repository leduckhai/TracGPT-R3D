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
class RawVisionTrainer(Trainer):
    def __init__(self, *args, eval_metrics=None, log_gradients=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_predictions = defaultdict(list)
        self.current_step = 0
        self.gradient_log_freq = 15
        self.log_gradients = log_gradients
        self.eval_metrics = eval_metrics or ['bbox_loss', 'pos_loss', 'size_loss', 'conf_loss']
        self.gradient_path = "gradient/gradient_logs.txt"
        self.metric_tracker=MetricTracker()
        self.eval_metric_trakcer=MetricTracker()
        self.loss= nn.CrossEntropyLoss()

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
        bbox_GCA_pred=subtask_out[:,0,:]
        bbox_Koedam_pred=subtask_out[:,1,:]
        bbox_MTA_pred=subtask_out[:,2,:]
        status_pred=finaltask_out
        loss,metric_loss = self.compute_model_loss( bbox_GCA,bbox_Koedam,bbox_MTA,status_criteria,outputs)     
        subtask_out,final_task_out=outputs   
        for key, value in metric_loss.items():
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)
        
        bbox_GCA_metric=self.metric_tracker.calculate_metrics_bbox_criteria("GCA",bbox_GCA_pred,bbox_GCA)
        bbox_Koedam_metric=self.metric_tracker.calculate_metrics_bbox_criteria("Koedam",bbox_Koedam_pred,bbox_Koedam)
        bbox_MTA_metric=self.metric_tracker.calculate_metrics_bbox_criteria("MTA",bbox_MTA_pred,bbox_MTA)
        bbox_status_metric=self.metric_tracker.calculate_metrics_bbox_criteria("Status",final_task_out,status_criteria)
        print("bbox GCA metric",bbox_GCA_metric)
        print("bbox Koedam metric",bbox_Koedam_metric)
        print("bbox MTA metric",bbox_MTA_metric)
        print("bbox status metric",bbox_status_metric)
        loss.backward()     
        # if self.state.global_step % self.gradient_log_freq == 0:
        #     self._log_gradient_stats(model, self.state.global_step)
        # gradient_anomaly = self._check_gradients()
        # self._clip_gradients()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
                
        return loss.detach()
 
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
        model = self.model.eval()
        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                bbox_criteria=inputs.get("bbox_criteria")
                status_criteria=inputs.get("status_criteria")
                image=inputs.get("images").to(model.device)
                outputs=model(image, bbox_criteria, status_criteria)
            
                loss,metric_loss = self.compute_model_loss( inputs,outputs)     
                subtask_out,final_task_out=outputs   
                print("metric loss",metric_loss)
                bbox_metric=self.eval_metric_trakcer.calculate_metrics_bbox_criteria(subtask_out,bbox_criteria)
                print("bbox metric",bbox_metric)
                status_metric=self.eval_metric_trakcer.calculate_metric_status(final_task_out,status_criteria)
                print("status metric",status_metric) 

        return 

    def log(self, metric,value,**kwargs):
        print("metric",metric,"value",value.item())
    # def log(self, logs, start_time=None, **kwargs):
        """
        Enhanced logging with auxiliary losses
        """
        # if hasattr(self, '_current_outputs'):
        #     for key, value in self._current_outputs.items():
        #         if key.endswith('_loss') and torch.is_tensor(value):
        #             logs[key] = value.item()
        
        # super().log(logs, start_time=start_time, **kwargs)
