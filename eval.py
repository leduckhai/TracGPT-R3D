from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v3")
# train_ds=ds["train"]

# A2_vals=train_ds.iloc[ "Patient ID"]
from bert_score import score
from uuid import uuid4
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import torch
import yaml
from utils.type import dict_to_namespace
# from model.bbox3d.bbox_head import BBox3DHead
from model.bbox3d.builder import BBox3DPredictor
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from datetime import datetime

now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d++%H:%M:%S")

config_path="config/llama.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
config=dict_to_namespace(config)

predictor=BBox3DPredictor(config.tiny_llama.bbox_predictor)

def evaluate(model,data_loader,tokenizer,save_path,save_bbox=True):
    id=date_time_str 
    bert_metrics_detail=[]
    metrics_all={}
    save_bbox_data=[]
    save_dir=os.path.join(save_path,id)
    os.makedirs(save_dir,exist_ok=True)
    metrics_all_path=os.path.join(save_path,id, "metrics_all.json")
    metrics_detail_path=os.path.join(save_path,id, "metrics_detail.json")
    save_bbox_path=os.path.join(save_path,id, "bbox_pred.json")

    precisions=[]
    recalls=[]
    f1_scores=[]
    iou_scores=[]
    for i, batch in enumerate(data_loader):
        (
            images,
            input_ids,
            attention_mask,
            labels,
            bbox_gt,
            bbox_mask,
            position_ids,
            answer_types,
            questions,
            answers
        ) = batch.values()
        images = images.to("cuda")
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        labels = labels.to("cuda")
        bbox_gt = bbox_gt.to("cuda")
        bbox_mask = bbox_mask.to("cuda")
        position_ids = position_ids.to("cuda")
        with torch.no_grad():
            outputs,bbox_preds = model.generate(input_ids=input_ids, images=images)
        generated_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        print("bbox_preds",bbox_preds["bbox_pred"])
        bbox_preds=bbox_preds["bbox_pred"]
        for i, text in enumerate(generated_text):
            if answer_types[i] in ["bbox_2d", "bbox_3d"]:
                bbox_output=predictor.compute_ious(bbox_preds[i], bbox_gt[i], bbox_mask[i])
                if bbox_output is None:
                    continue
                bbox_output["pred"]=[pred.detach().cpu().numpy() for pred in bbox_output["pred"]]
                bbox_output["gt"]=[gt.detach().cpu().numpy() for gt in bbox_output["gt"]]
                bbox_output["iou"]=[iou.detach().cpu().numpy() for iou in bbox_output["iou"]]
                for iou in bbox_output["iou"]:
                    iou_scores.append(iou)
                save_bbox_data.append(bbox_output)
                print("bbox output sample",bbox_output)
                save_bbox_data.append(bbox_output)
            # print("type text",answers[i], "text is",text, type(text))
            P, R, F1 = score([answers[i]],  [text], lang="en", model_type="bert-base-uncased")

            print(f"Precision: {P.mean().item():.4f}")
            print(f"Recall:    {R.mean().item():.4f}")
            print(f"F1:        {F1.mean().item():.4f}")
            bert_metrics_detail.append({
                "answer":answers[i],
                "text":text,
                "precision":P.mean().item(),
                "recall":R.mean().item(),
                "f1":F1.mean().item()
            })
            precisions.append(P.mean().item() )
            recalls.append(R.mean().item())
            f1_scores.append(F1.mean().item())

    metrics_all["precisions"]=np.mean(precisions)
    metrics_all["recall"]=np.mean(recalls)
    metrics_all["f1_score"]=np.mean(f1_scores)
    metrics_all["iou"]=np.mean(iou_scores)
    with open(metrics_all_path,"w") as f:
        f.write(str(metrics_all))
    with open(metrics_detail_path,"w") as f:
        f.write(str(bert_metrics_detail))
    with open(save_bbox_path,"w") as f:
        f.write(str(save_bbox_data))

    print("bert metrics detail",bert_metrics_detail)
    print("bert metrics all",metrics_all)
    
    