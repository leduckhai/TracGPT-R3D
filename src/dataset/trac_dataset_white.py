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
from functools import lru_cache

class TracDatasetWhite(Dataset):
    def __init__(self, data_paths, image_path, mode="train", n_sample=-1, image_shape=[32, 256, 256], dataset_config=None):
        self.image_shape = image_shape
        self.only_2d = dataset_config.get("only_2d", False) if dataset_config else False
        # self.base_transform = base_transform_2d if self.only_2d else base_transform_3d
        self.base_transform=base_transform_3d
        self.img_dir = image_path
        self.sample_indices = []
        self.mode=mode
        # Load metadata only
        for path in data_paths:
            with open(path, "r") as f:
                data = json.load(f)
                self.sample_indices.extend([(path, i) for i in range(len(data))])
        
        if n_sample > 0:
            self.sample_indices = self.sample_indices[:n_sample]
        random.shuffle(self.sample_indices)

    # @lru_cache(maxsize=1000)
    def _load_image(self, patient_id, slice_order):
        image_paths = [os.path.join(self.img_dir, patient_id, f"{s}.pkl") for s in slice_order]
        return convert_list_slice_paths_to_3d(image_paths)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        path, sample_idx = self.sample_indices[idx]
        with open(path, "r") as f:
            data_point = json.load(f)[sample_idx]
        # print("Loading data point:", data_point)
        image = self._load_image(data_point["Patient ID"], data_point["slice order"])
        image = self.base_transform({"image": image})["image"].squeeze(0)
        if self.mode!="train":
            status=""
        else:
            status = data_point["A4"]
            
        return {
            "image": image,
            "P_ID": data_point["Patient ID"],
            "Q1": data_point["Q1"],
            "Q2": data_point["Q2"],
            "Q3": data_point["Q3"],
            "Q4": data_point["Q4"],
            "A1": data_point["A1"],
            "A2": data_point["A2"],
            "A3": data_point["A3"],
            "A4": data_point["A4"],
            "answer": status,
        }
if __name__ == "__main__":
    train_dirs="pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/train/data"
    test_dir= "pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/test/data"
    path=[os.path.join(train_dirs, f) for f in os.listdir(train_dirs) if f.endswith('.json')]
    test_path=[os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')]
    for f in path:
        with open(f, "r") as file:
            data = json.load(file)
            print(f"Loaded {len(data)} samples from {f}")
            
    for f in test_path:
        with open(f, "r") as file:
            data = json.load(file)
            print(f"Loaded {len(data)} samples from {f}")
