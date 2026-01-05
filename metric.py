from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score
from collections import defaultdict
gt_labels={
    "Non-Dementia": 0,
    "Mild-Dementia": 1, 
    "Moderate-Dementia": 2, 
}

pred_labels={
    "non": 0,  
    "mild": 1, 
    "moderate": 2
}

def calculate_metric(preds, labels):
    for label in labels:
        if label not in gt_labels:
            print("New GT label found:", label)
            # gt_labels[label]=len(gt_labels)
    gt_mapped = [gt_labels.get(label, -1) for label in labels]
    
    # if -1 in gt_mapped: 
    #     raise ValueError("Some ground truth labels are not recognized.")    
    pred_mapped=[]
    for pred in preds:
        found=False
        for key in pred_labels.keys():
            if key in pred.lower():
                pred_mapped.append(pred_labels[key])
                found=True
                break
        if not found:
            print("New Pred label found:", pred)
            pred_mapped.append(-1) 
    gt_counter={}
    pred_counter={}
    for g in gt_mapped:
        if g not in gt_counter:
            gt_counter[g]=0
        gt_counter[g]+=1
    for p in pred_mapped:   
        if p not in pred_counter:
            pred_counter[p]=0
        pred_counter[p]+=1
    print("GT distribution:", gt_counter)
    print("Pred distribution:", pred_counter)
    if -1 in pred_mapped: 
        raise ValueError("Some predicted labels are not recognized.")   
    
    precision, recall, f1, support = precision_recall_fscore_support(
    gt_mapped, pred_mapped, labels=[0,1,2], average=None
)

    for i, cls in enumerate([0,1,2]):
        print(f"Class {cls}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")
    macro_precision = precision_score(gt_mapped, pred_mapped, average='macro')
    macro_recall    = recall_score(gt_mapped, pred_mapped, average='macro')
    macro_f1        = f1_score(gt_mapped, pred_mapped, average='macro')

    # Micro-average (global counts)
    micro_precision = precision_score(gt_mapped, pred_mapped, average='micro')
    micro_recall    = recall_score(gt_mapped, pred_mapped, average='micro')
    micro_f1        = f1_score(gt_mapped, pred_mapped, average='micro')

    print("\nOverall metrics:")
    print(f"Macro Precision={macro_precision:.3f}, Macro Recall={macro_recall:.3f}, Macro F1={macro_f1:.3f}")
    print(f"Micro Precision={micro_precision:.3f}, Micro Recall={micro_recall:.3f}, Micro F1={micro_f1:.3f}")  
if __name__ == "__main__":
    result_path="full_eval/ocgq0zoy_test/inference_results.json"
    import json
    with open(result_path, "r") as f:
        data = json.load(f)
    preds = data["predictions"]
    labels = data["references"]
    calculate_metric(preds, labels)
    
    
    # gt {1: 85, 0: 189, 2: 185}
    # Pred distribution: {1: 264, 0: 180, 2: 15}
    
    # Class 0: Precision=0.867, Recall=0.825, F1=0.846, Support=189
    # Class 1: Precision=0.277, Recall=0.859, F1=0.418, Support=85
    # Class 2: Precision=0.000, Recall=0.000, F1=0.000, Support=185
