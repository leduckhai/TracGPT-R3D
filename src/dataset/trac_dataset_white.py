import monai
original_set_random_state = monai.transforms.compose.Compose.set_random_state

from torch.utils.data import Dataset
import os
import json
import sys
import os
from dotenv import load_dotenv
import random

load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
from src.data_process.util import convert_list_slice_paths_to_3d
import os
import numpy as np
import json
from collections import defaultdict
from monai.transforms import Compose
import torch.nn.functional as F
from monai.transforms import Compose, ResizeD, EnsureChannelFirstD,ScaleIntensityRanged

def get_base_transform(spatial_size=[32, 256, 256]):
    return Compose(
            [
                EnsureChannelFirstD(keys=["image"], channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=255,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ResizeD(
                    keys=["image"],
                    spatial_size=spatial_size,
                    mode="trilinear",
                    size_mode="all",
                ),
            ]
        )

class TracDatasetWhite(Dataset):
    def __init__(self, data_paths, image_path, mode="train", n_sample=-1,  balance_A4=True,is_transform=True):
        self.img_dir = image_path
        self.sample_indices = []
        self.mode = mode
        self.is_transform=is_transform
        # self.train_transform = train_transform if mode == "train" else None
        self.max_depth=64
        self.height=256
        self.width=256
        all_data = []
        random.seed(68)   
        for path in data_paths:
            with open(path, "r") as f:
                data = json.load(f)
                for i, dp in enumerate(data):
                    a4 = dp["A4"].lower().replace("-", " ")
                    all_data.append((path, i, a4))

        random.shuffle(all_data)
        if n_sample > 0:
            all_data = all_data[:n_sample]
        self.sample_indices=all_data   
        if self.mode=="train":
            counter = defaultdict(int)
            for _, _, a4 in all_data:
                counter[a4] += 1
            print("A4 distribution before balancing:", dict(counter))

            by_a4 = defaultdict(list)
            for item in all_data:
                by_a4[item[2]].append(item)

            max_count = max(len(v) for v in by_a4.values())

            # ---- UPSAMPLE ----
            balanced_data = []
            for a4, items in by_a4.items():
                if len(items) < max_count:
                    items = items + random.choices(items, k=max_count - len(items))
                balanced_data.extend(items)

            random.shuffle(balanced_data)
            self.sample_indices = balanced_data
        print("Length of data:", len(self.sample_indices))
        counter = defaultdict(int)
        for _, _, a4 in self.sample_indices:
            counter[a4] += 1
        print("A4 distribution after balancing:", dict(counter))
        self.map={
            "non":0,
            "mild":1,
            "moderate":2,
        }
        self.base_transform= get_base_transform(spatial_size=[-1, self.height, self.width])

    def _load_image(self, patient_id, slice_order):
        image_paths = [os.path.join(self.img_dir, patient_id, f"{s}.pkl") for s in slice_order]
        image =convert_list_slice_paths_to_3d(image_paths)
        return image

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        path, sample_idx,_ = self.sample_indices[idx]
        with open(path, "r") as f:
            data_point = json.load(f)[sample_idx]
        image = self._load_image(data_point["Patient ID"], data_point["slice order"])
        answer = ""
        if self.mode !="test":
            answer = data_point["A4"].lower().replace("-", " ").split(" ")[0]
            
        if self.is_transform:
            image = self.base_transform({"image": image})["image"]
            # if self.train_transform:
            #     image = self.train_transform({"image": image})["image"]
            #     image=image.squeeze(0)
            
        # idx=random.randrange(len(data_point["Q4"]))
        question="How grievous is this medical situation?"
        # for key,value in self.map.items():
        #     if key.lower() in data_point["A4"].lower():
        #         label_idx=value
        #         break
        # else:
        #     label_idx=-1
        return {
            "image": image,
            "slice_order": data_point["slice order"],
            "P_ID": data_point["Patient ID"],
            "question":question,
            "Q1": data_point["Q1"],
            "Q2": data_point["Q2"],
            "Q3": data_point["Q3"],
            "Q4": data_point["Q4"],
            "A1": data_point["A1"],
            "A2": data_point["A2"],
            "A3": data_point["A3"],
            "A4": data_point["A4"],
            "answer": answer,
            # "label_idx": label_idx
        }
        
if __name__ == "__main__":
    import yaml
    from transformers import AutoTokenizer
    train_val_dir= "pseudo_3d/32_all_slices/0fa350fe-9eb2-4b61-916f-5a29e322f6ab/train/data"
    test_dir= "pseudo_3d/32_all_slices/0fa350fe-9eb2-4b61-916f-5a29e322f6ab/test/data"
    train_image="clean_data_s_chain/train/image"
    test_path=[os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')]
    train_path=[os.path.join(train_val_dir, f) for f in os.listdir(train_val_dir) if f.endswith('.json')]
    dataset = TracDatasetWhite( train_path, train_image, mode="train", n_sample=-1,  balance_A4=True)
    # 6000 sample, 6000/8=750 steps per epoch with batch size 8, max step=1.3*750=975 steos
    for i in range(3):
        sample = dataset[i]
        print("image",sample["image"].shape)
        # print(sample["image"].shape, sample["P_ID"], sample["A4"], sample["Q1"], sample["A1"])
  