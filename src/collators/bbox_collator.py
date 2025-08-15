import torch 
from torch.utils.data import Dataset
import numpy as np
import sys 
sys.path.append("/root/TracGPT-R3D")
from collections import defaultdict


status_map={
    "Status (Non-Dementia)":0,
    "Status (Mild-Dementia)":1,
    "Status (Moderate-Dementia)":2
}


class BboxCollator:
    def __init__(self,):
        # print("BboxCollator initialized with width:", width, "and height:", height)
        width=256
        height=256
        self.width = width
        self.height = height
    def process_bboxes(self, batch_bboxes,tensor=False):
        output_bboxes = []
        for bboxes in batch_bboxes:
            if not tensor:
                bboxes = [[bbox[0]*self.width, bbox[1]*self.height, bbox[2]*self.width, bbox[3]*self.height]
                            for bbox in bboxes]
            else:
                bboxes = torch.tensor([[bbox[0]*self.width, bbox[1]*self.height, bbox[2]*self.width, bbox[3]*self.height]
                            for bbox in bboxes]).float()

            # print("processed bboxes", len(bboxes),bboxes )
            output_bboxes.append(bboxes)
        return output_bboxes
    def __call__(self, batch):
        images = []
        bbox_metrics = defaultdict(list)
        status_targets = []
        batch_bboxes= []
        for sample in batch:
            images.append(sample['image'].detach().clone())
            status_targets.append(status_map[sample['A4']])
            for metric, val in sample["A3"].items():
                bbox_metrics[metric].append(val)
            processed_bboxes = self.process_bboxes(sample["A1"])
            batch_bboxes.append(processed_bboxes)
        processed = {
            "images":images,
            "bboxes": batch_bboxes,  # [B, num_bboxes, 4]
            "status_criteria": torch.tensor(status_targets).long()  # [B]
        }
        
        for metric, values in bbox_metrics.items():
            processed[f"bbox_{metric}"] = torch.tensor(values).float()  # [B, ...]
        
        return processed


if __name__ == "__main__":
    from src.dataset.dataloader import load_data
    import torch
    from torch.utils.data import Dataset, DataLoader 
    
    collator=BboxCollator()
    # train_set, val_set, test_set = load_data(train_val_dir="/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/1ad40bdc-da39-4dd8-9d3a-27ce00e754fa/train/data",dataset="trac_white",train_sample=-1,val_sample=-1,test_sample=-1)
    train_set, val_set, test_set = load_data(train_val_dir="/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/1ad40bdc-da39-4dd8-9d3a-27ce00e754fa/train/data",dataset="trac_white")
    print("len train set",len(train_set), "len val set",len(val_set),"len test_set",len(test_set))
    train_ld=DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collator)
    for i, sample in enumerate(train_ld):
        
        if i==3:
            break
        # print("sample",sample)
        batch_bboxes=sample["bboxes"]
        image=sample["images"]
        print("image shape", image.shape)
        # bboxes: B, N slices per 3d image(32), n_bboxes, 4
        # print("len bboxes",len(batch_bboxes),len(batch_bboxes[0]),batch_bboxes[0])
        # print("sample",sample.keys(),sample["bbox_criteria"],sample["status_criteria"])
        # print("sample",sample["bbox_GCA"],sample["bbox_Koedam"],sample["bbox_MTA"],sample["status_criteria"])
        