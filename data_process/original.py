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
import uuid
uuid=str(uuid.uuid4())
ds = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v3")
output_folder="./clean_data_v2"
drop_keys=["image","image_with_bboxes","__index__level_0__","No.", "Column 9","Doctor","Start date","Google Drive Link","rotated_link","vn","fr","de","mandarin","korean","japanese","vi"]

for key in ds.keys():
    subdir=f"{output_folder}/{uuid}/{key}"
    os.makedirs(subdir, exist_ok=True)
    original_group=defaultdict(list)
    print("key", key)
    for i in tqdm(range(len(ds[key]))):
        sample=ds[key][i]
        for k in drop_keys:
            if k in sample:
                del sample[k]
        original_prop=sample["Original"]
        
        original_group[original_prop].append(sample)

    counter=0
    for original_prop,ls in original_group.items():
        print("original_prop",original_prop, len(ls))

        output_file=f"{subdir}/prop_{counter}.json"
        counter+=1
        with open(output_file, "w") as f:
            json.dump(ls, f)
        
       
