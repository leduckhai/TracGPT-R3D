import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json 
import pickle

class TracDataset(Dataset):
    def __init__(self, mode,patient_records,root_dir, transform=None):
        self.mode = mode
        self.patient_records = patient_records
        self.root_dir = root_dir
        self.slice_records=[]
        for record in patient_records:
            path=os.os.path.join(root_dir,"data",record)
            with open(path,"r") as f:
                data=json.load(f)
                self.slice_records.append(data)
               
    def __len__(self):
        return len(self.slice_records)

    def __getitem__(self, idx):
        slide_record=self.slice_records[idx]
        image_path = os.path.join(self.root_dir, "image", f'{slide_record["Patient ID"]}.pkl')
        with open(image_path, "rb") as f:
            image = pickle.load(f)
        
        q1,a1,q2,a2,q3,a3,q4,a4=slide_record["Q1"],slide_record["A1"],slide_record["Q2"],slide_record["A2"],slide_record["Q3"],slide_record["A3"],slide_record["Q4"],slide_record["A4"]
        index=slide_record["slide_idx"]
        return image,q1,a1,q2,a2,q3,a3,q4,a4,index
if __name__=="__main__":
    import os
    import random
    from sklearn.model_selection import train_test_split
    train_val_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data_3d/50_slices/5b553c54-e609-4172-86e7-733c774ac83dtrain"
    patient_records=os.listdir(os.path.join(train_val_dir,"data"))
    patient_records = sorted(patient_records) 
    print("num records",len(patient_records))
    train_records, val_records = train_test_split(patient_records, test_size=0.2, random_state=42)
    train_set=TracDataset("train",train_records,train_val_dir)
