import numpy as np
from typing import Dict, List, Union
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from collections import defaultdict

class MetricTracker:
    def __init__(self):
        self.global_bbox_metrics = defaultdict(list)
        self.global_status_metrics=defaultdict(list)
        self.global_metrics={}
    def get_global_bbox_metrics(self):
        return self.global_metrics
    

        
    def calculate_metrics_bbox_criteria(
        self,
        metric_name: str,
        y_pred: torch.Tensor,  
        y_true: torch.Tensor,
        average: str = 'macro'
    ) -> Dict[str, Dict[str, float]]:
        y_true_numpy = y_true.detach().long().cpu().numpy().flatten()
    
        # Handle predictions
        if y_pred.ndim == 2:  # Logits/probabilities
            y_pred_numpy = torch.argmax(y_pred, dim=-1).cpu().numpy()
        else:  # Already class indices
            y_pred_numpy = y_pred.detach().cpu().numpy()
        # print("y_true numpy",y_true_numpy,"y_pred numpy",y_pred_numpy)
        n_classes = len(np.unique(y_true_numpy))
        avg_method =  average
        
        metrics = {
            'accuracy': accuracy_score(y_true_numpy, y_pred_numpy),
            'precision': precision_score(y_true_numpy, y_pred_numpy, 
                                    average=avg_method, zero_division=0),
            'recall': recall_score(y_true_numpy, y_pred_numpy,
                                average=avg_method, zero_division=0),
            'f1': f1_score(y_true_numpy, y_pred_numpy,
                        average=avg_method, zero_division=0),
        }
        
        # Update global metrics (fixed iteration)
        for key, value in metrics.items():
            if metric_name not in self.global_metrics:
                self.global_metrics[metric_name] ={}
            if key not in self.global_metrics[metric_name]:
                self.global_metrics[metric_name][key] = []
            self.global_metrics[metric_name][key].append(value)
        
        return metrics
