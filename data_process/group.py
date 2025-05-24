from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os 
import numpy as np
import pandas as pd
import json
import pickle 
import shutil


ds = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v3")

base_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/"
train_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train"
test_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean/test"

p_ids = set()

for key in ds.keys():  
    save_folder = train_dir if key == "train" else test_dir
    for i in tqdm(range(len(ds[key])), desc=f"Processing {key}"):
        sample = ds[key][i]
        pid = sample["Patient ID"]

        if pid not in p_ids:
            p_ids.add(pid)
            src_path = os.path.join(base_dir, pid)
            dst_path = os.path.join(save_folder, pid)
            if os.path.exists(src_path):  # add safety check
                shutil.move(src_path, dst_path)
            else:
                print(f"Warning: {src_path} does not exist.")