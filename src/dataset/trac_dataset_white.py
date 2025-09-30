from monai.utils.misc import MAX_SEED
import monai
# Backup original method
original_set_random_state = monai.transforms.compose.Compose.set_random_state

def patched_set_random_state(self, seed=None):
    if seed is None:
        seed = np.random.randint(0, MAX_SEED, dtype=np.uint32)
    else:
        seed = int(seed) % MAX_SEED  # Ensure within bounds
    
    self.R = np.random.RandomState(seed)
    for transform in self.transforms:
        if hasattr(transform, 'set_random_state'):
            # Generate safe random seed for each transform
            safe_seed = int(self.R.randint(0, MAX_SEED))
            transform.set_random_state(seed=safe_seed)

# Apply the patch
monai.transforms.compose.Compose.set_random_state = patched_set_random_state
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
from src.dataset.transform import base_transform_3d, train_transform
from functools import lru_cache
from collections import defaultdict
# import torchio as tio
class TracDatasetWhite(Dataset):
    def __init__(self, data_paths, image_path, mode="train", n_sample=-1, image_shape=[32, 256, 256], dataset_config=None, balance_A4=True):
        self.image_shape = image_shape
        self.base_transform = base_transform_3d
        self.img_dir = image_path
        self.sample_indices = []
        self.mode = mode
        self.train_transform = train_transform if mode == "train" else None
        random.seed(68)   
        all_data = []
        print("mode", mode)
        for path in data_paths:
            with open(path, "r") as f:
                data = json.load(f)
                for i, dp in enumerate(data):
                    all_data.append((path, i, dp["A4"].lower().replace("-", " ")))
        random.shuffle(all_data)
        counter=defaultdict(int)
        if n_sample > 0:
            all_data = all_data[:n_sample]
        for _,_,a4 in all_data:
            counter[a4]+=1
        print("A4 distribution before balancing:", dict(counter))

        if balance_A4 and mode == "train":
            print("Balancing dataset based on A4 labels")
            buckets = defaultdict(list)
            for x in all_data:
                buckets[x[2]].append(x)

            max_len = max(len(v) for v in buckets.values())
            print("Balancing to max bucket size:", max_len)
            balanced = []
            for label, items in buckets.items():
                balanced.extend(random.choices(items, k=max_len))
            self.sample_indices = [(p, i,a) for p, i, a in balanced]
        else:
            self.sample_indices = [(p, i,a) for p, i, a in all_data]
        counter=defaultdict(int)
        random.shuffle(self.sample_indices)
        for _,_,a4 in self.sample_indices:
            counter[a4]+=1
        print("A4 distribution after balancing:", dict(counter))


    def _load_image(self, patient_id, slice_order):
        image_paths = [os.path.join(self.img_dir, patient_id, f"{s}.pkl") for s in slice_order]
        return convert_list_slice_paths_to_3d(image_paths)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        path, sample_idx,_ = self.sample_indices[idx]
        with open(path, "r") as f:
            data_point = json.load(f)[sample_idx]
        image = self._load_image(data_point["Patient ID"], data_point["slice order"])
        # image = np.expand_dims(image, axis=0)    
        image = self.base_transform({"image": image})["image"]
        # image = transformed_subject.image.data 
        # print("image shape", image.shape)
        status = ""
        if self.mode == "train":
            pretext="The diagnosis is"
            status = data_point["A4"].lower().replace("-", " ")
            status = f"{pretext} {status}."
        if self.train_transform:
            image = self.train_transform({"image": image})["image"]
            image=image.squeeze(0)
            
        idx=random.randrange(len(data_point["Q4"]))
        question = data_point["Q4"][idx]
        return {
            "image": image,
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
            "answer": status,
        }
        
if __name__ == "__main__":
    import yaml
    from transformers import AutoTokenizer
    # train_dirs="pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/train/data"
    # test_dir= "pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/test/data"
    # path=[os.path.join(train_dirs, f) for f in os.listdir(train_dirs) if f.endswith('.json')]
    # test_path=[os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')]
    # for f in path:
    #     with open(f, "r") as file:
    #         data = json.load(file)
    #         print(f"Loaded {len(data)} samples from {f}")
            
    # for f in test_path:
    #     with open(f, "r") as file:
    #         data = json.load(file)
    #         print(f"Loaded {len(data)} samples from {f}")
    # dataset = TracDatasetWhite( path, "/root/VLMTrac/2d_data/train/image", mode="train", n_sample=-1, image_shape=[32, 256, 256], dataset_config=None, balance_A4=True)
    # for i in range(3):
    #     sample = dataset[i]
    #     print(sample["image"].shape, sample["P_ID"], sample["A4"], sample["Q1"], sample["A1"])
  