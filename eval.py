from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v3")
# train_ds=ds["train"]

# A2_vals=train_ds.iloc[ "Patient ID"]
from bert_score import score
from uuid import uuid4
import os
import pandas as pd
import numpy as np
import torch
import yaml
from utils.type import dict_to_namespace
# from model.bbox3d.builder import BBox3DPredictor
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from datetime import datetime
from model.bbox3d.helper import compute_ious
now = datetime.now()
date_time_str = now.strftime("%Y-%m-%d++%H:%M:%S")
from tqdm import tqdm


def evaluate_single(bbox_pred,bbox_gt,bbox_mask):
    print("bbox mask",bbox_mask)
    if torch.any(bbox_mask):
        print("got mask")
        bbox_output=compute_ious(bbox_pred, bbox_gt, bbox_mask)
        preds=[pred.detach().cpu().numpy() for pred in bbox_output["pred"]]
        gt=[label.detach().cpu().numpy() for label in bbox_output["gt"]]
        iou=[iou.detach().cpu().numpy() for iou in bbox_output["iou"]]
        return preds,gt,iou
    return [],[],[]

def evaluate(model,data_loader,tokenizer,save_path,save_bbox=True,skip_text_question=False):
    id=date_time_str 
    bert_metrics_detail=[]
    metrics_all={}
    save_bbox_data=[]
    save_dir=os.path.join(save_path,id)
    print("save dir evaluate",save_dir)
    os.makedirs(save_dir,exist_ok=True)
    metrics_all_path=os.path.join(save_path,id, "metrics_all.json")
    metrics_detail_path=os.path.join(save_path,id, "metrics_detail.json")
    save_bbox_path=os.path.join(save_path,id, "bbox_pred.json")

    precisions=[]
    recalls=[]
    f1_scores=[]
    iou_scores=[]
    for i, batch in enumerate(tqdm(data_loader, desc="Processing Batches")):
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
        if skip_text_question and "bbox_3d" not in answer_types:
            print("skip text question")
            continue
            # outputs = model.generate(input_ids=input_ids, images=images)
        with torch.no_grad():
            outputs,bbox_preds = model.generate(input_ids=input_ids, images=images)
        generated_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        bbox_preds=bbox_preds["bbox_pred"]
        for i, text in enumerate(generated_text):
            if answer_types[i] in ["bbox_2d", "bbox_3d"]:

                pred,gt,iou=evaluate_single(bbox_preds[i],bbox_gt[i],bbox_mask[i])
                if len(pred):
                    iou_scores.extend(iou)
                    print("mean iou sample",np.mean(iou))
                    bbox_output={"pred":pred,"gt":gt,"iou":iou}
                    save_bbox_data.append(bbox_output)
                    # print("bbox output sample",bbox_output)
                    save_bbox_data.append(bbox_output)
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
    
    