import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import pickle
import sys
import os
from dotenv import load_dotenv
from types import SimpleNamespace
load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
from transformers import AutoTokenizer
from data_process.util import convert_list_slice_paths_to_3d
import monai.transforms as mtf
import random
import os
import numpy as np
import json
from monai.transforms import Compose, ResizeD
from typing import Mapping, Hashable
from data.transform import  ResizeBBox3D
from collections import defaultdict

class TracDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        mode="train",
        root_dir="/root/VLMTrac",
        args=None,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.base_transform = Compose(
            [
                ResizeD(keys=["image"], spatial_size=[32, 256, 256], mode="bilinear"),
                ResizeBBox3D(
                    keys=["bboxes"], orig_size=[50, 256, 412], target_size=[32, 256, 256]
                ),
            ]
        )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )
        data_dir=os.path.join(root_dir,"chunks",mode, "data")
        self.img_dir = os.path.join(root_dir, "2d_data",mode,"image")
        assert os.path.exists(data_dir) , f"{data_dir} does not exist"
        assert os.path.exists(self.img_dir) , f"{self.img_dir} does not exist"
        
        data_paths = [
            os.path.join(data_dir, record) for record in os.listdir(data_dir)
        ]

        self.qa_banks=[]
        qa_maps={
            "Q1":"A1",
            "Q2":"A2",
            "Q3":"A3",
            "Q4":"A4",
        }
        for path in data_paths:
            with open(path, "r") as f:
                data = json.load(f)

            for sample in data:
                for q,a in qa_maps.items():
                    data_point={
                        'slice order':sample['slice order'],
                        'Patient ID':sample['Patient ID'],
                        "question":sample[q][0],
                        "answer":sample[a],
                    }
                    if q=="Q1":
                        data_point["answer_type"]='bbox_3d'
                        data_point["bbox_3d"]=sample[a]
                    else:
                        data_point["answer_type"]='text'
                        data_point["bbox_3d"]=None
                    self.qa_banks.append(data_point)
            
            
    def __len__(self):
        return len(self.qa_banks)
   
    def __getitem__(self, idx):
        data_point = self.qa_banks[idx]
        slice_order = data_point["slice order"]
        patient_id = data_point["Patient ID"]

        image_path = [
            os.path.join(self.img_dir, patient_id, f"{s}.pkl")
            for s in slice_order
        ]
        print("image_path",image_path[0])
        for path in image_path:
            assert os.path.exists(path) , f"{path} does not exist"
        image_3d = convert_list_slice_paths_to_3d(image_path)

        data_point["image"] = image_3d
        return data_point



if __name__ == "__main__":
    import os
    import random
    from sklearn.model_selection import train_test_split
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    # train_val_dir = "/home/ubuntu/repo/TracGPT-R3D/VLMTrac/50_chunk_data/train"
    # patient_records = os.listdir(os.path.join(train_val_dir, "data"))
    # patient_records = sorted(patient_records)
    # train_records, val_records = train_test_split(
    #     patient_records, test_size=0.2, random_state=42
    # )


    # train_set = TracDataset( tokenizer=tokenizer,mode="train")
    # for i in range(len(train_set)):
    #     print(train_set[i])

    sample_dirs="/root/VLMTrac/2d_data/train/image/OAS1_0001"
    files=[os.path.join(sample_dirs,f) for f in os.listdir(sample_dirs)]
    
    def is_pickle_header(path):
        with open(path, 'rb') as f:
            first_bytes = f.read(2)
            print("first_bytes",first_bytes)
            return first_bytes == b'\x80\x04'  # pickle protocol 4

    print("valid",is_pickle_header(files[0]))
    sample = convert_list_slice_paths_to_3d(files)
    print(sample.shape)
