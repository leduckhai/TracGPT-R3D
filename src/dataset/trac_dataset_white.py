import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import pickle
import sys
import os
from dotenv import load_dotenv
import random
from sklearn.model_selection import train_test_split

load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
from src.data_process.util import convert_list_slice_paths_to_3d
import os
import numpy as np
import json
from src.dataset.transform import base_transform_3d, base_transform_2d

class TracDatasetWhite(Dataset):
    def __init__(
        self,
        data_paths,
        image_path,
        mode="train",
        n_sample=-1,
        image_shape=[32, 256, 256],
        dataset_config: dict = {},
    ):
        only_2d = dataset_config.get("only_2d", False)
        self.image_shape = image_shape
        self.only_2d = only_2d
        self.mode = mode
       
        if self.only_2d:
            self.base_transform = base_transform_2d
        else:
            self.base_transform = base_transform_3d

      
        self.img_dir = image_path

        self.qa_banks = []
     
        for path in data_paths:
            
            with open(path, "r") as f:
                data = json.load(f)

            for sample in data:
                data_point = {
                    "slice_order": sample["slice order"],
                    "Patient_ID": sample["Patient ID"],
                    "Q1": sample["Q1"],
                    "Q2": sample["Q2"],
                    "Q3": sample["Q3"],
                    "Q4": sample["Q4"],
                    "A1": sample["A1"],
                    "A2": sample["A2"],
                    "A3": sample["A3"],
                    "A4": sample["A4"],
                }

                self.qa_banks.append(data_point)
       
        if n_sample != -1:
            self.qa_banks = self.qa_banks[:n_sample]
        random.shuffle(self.qa_banks)

    def __len__(self):
        return len(self.qa_banks)

    def __getitem__(self, idx):
        data_point = self.qa_banks[idx]
        slice_order = data_point["slice_order"]
        patient_id = data_point["Patient_ID"]

        image_path = [
            os.path.join(self.img_dir, patient_id, f"{s}.pkl") for s in slice_order
        ]
        for path in image_path:
            assert os.path.exists(path), f"{path} does not exist"
        image_3d = convert_list_slice_paths_to_3d(image_path)
        image_dict = self.base_transform({"image":image_3d})
        image_dict["image"] = image_dict["image"].squeeze(0)

        data_point["image"] = image_dict["image"]
        return data_point
