from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os 
import numpy as np
import pandas as pd
import json
import pickle 
import shutil
from collections import defaultdict
import json 

ds = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v6")
output_folder="./clean_data"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

for key in ds.keys():
    print("key", key)
    output_dir=f"{output_folder}/{key}"

    clients_map={}

    annot_root=f"{output_dir}/image_with_bboxes"
    os.makedirs(annot_root, exist_ok=True)
    image_root=f"{output_dir}/image" 
    # data_path=f"{output_dir}/data.csv"
    data_folder=f"{output_dir}/data"
    os.makedirs(data_folder, exist_ok=True)

    os.makedirs(image_root, exist_ok=True)   
    patients=defaultdict(list)
    for i in tqdm(range(len(ds[key]))):
        client={}
        sample=ds[key][i]
        patient_id=sample["Patient ID"]
        patient_dir= f"{output_dir}/{patient_id}" 
        slide=sample["Slide"]
        annot_folder=f"{annot_root}/{patient_id}"
        image_folder=f"{image_root}/{patient_id}"
        os.makedirs(annot_folder, exist_ok=True)
        os.makedirs(image_folder, exist_ok=True)
      
        image_path=f"{image_folder}/{slide}.pkl"
        annot_path=f"{annot_folder}/{slide}.pkl"
        img_arr= np.array(sample["image"])
        annot_arr= np.array(sample["image_with_bboxes"])
        with open(image_path, "wb") as f:
            pickle.dump(img_arr, f)
        with open(annot_path, "wb") as f:
            pickle.dump(annot_arr, f)
        del sample["image"]
        del sample["image_with_bboxes"]
        
        patients[patient_id].append(sample) 
    for patient_id in patients.keys():
        patient_data_path=f"{data_folder}/{patient_id}.json"
        
        
        with open(patient_data_path, "w", encoding="utf-8") as f:
            json.dump(patients[patient_id], f, ensure_ascii=False, indent=2)


        
       

file_1="/home/ducnguyen/sync_local/repo/TracGPT/vlm-project-with-images-with-bbox-images-v3/data/train-00001-of-00004.parquet"
