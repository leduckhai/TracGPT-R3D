import sys 
sys.path.append("/home/ubuntu/repo/TracGPT-R3D")
import json 
import shutil
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import pickle
from collections import Counter, defaultdict
from PIL import Image
from data_process.util import sort_files,save_nifti
import uuid
import nibabel as nib
import math
from data_process.util import group_and_merge_3d_bboxes_v2,convert_list_slice_paths_to_3d,draw_3d_bbox_wireframe_v2
import ast
drop_keys=["image","image_with_bboxes","Original","__index__level_0__","No.", "Column 9","Deliverable","Doctor","Start date","Google Drive Link","rotated_link","vn","fr","de","mandarin","korean","japanese","vi"]

dsc_path="data_process/desc.json"
with open(dsc_path,"r") as f:
    desc_map=json.load(f)

def test_concat_bbox():
    n_concat=50  
    sample_file="clean_data_3d/50_slices/3393c76c-3917-4791-83e7-e32936431012/train/data/OAS1_0056.json"
    
    img_annot_dir="/home/ubuntu/repo/TracGPT-R3D/clean_data/train/image_with_bboxes"

    with open(sample_file, "r") as f:
        data=json.load(f)
    slides=[data["Slide"] for data in data]
    slides=sort_files(slides)
    slide_map={}
    p_id=data[0]["Patient ID"]
    for d in data:
        for k in drop_keys:
            if k in d:
                del d[k]
        slide_map[d["Slide"]]=d
    sample=slides[:n_concat]
    list_bboxes=[slide_map[s]["A1"] for s in sample]
    bbox_input = []
    for bboxes_ls in list_bboxes:
        parsed = ast.literal_eval(bboxes_ls)
        bbox_input.extend(parsed)
    
    output=group_and_merge_3d_bboxes_v2(bbox_input)

    
    all_annot_paths=[os.path.join(img_annot_dir,p_id,f"{s}.pkl") for s in sample]
    with open(all_annot_paths[0],"rb") as f:
        slide=pickle.load(f)
    gt=convert_list_slice_paths_to_3d(all_annot_paths)
    shape=(n_concat,slide.shape[0],slide.shape[1])
    print("shape",shape)
    unormalize_bbox=[]
    for bbox in output:
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        x_min=x_min*shape[2]
        y_min=y_min*shape[1]
        z_min=z_min*shape[0]
        x_max=x_max*shape[2]
        y_max=y_max*shape[1]
        z_max=z_max*shape[0]
        unormalize_bbox.append([x_min, y_min, z_min, x_max, y_max, z_max])
    print("unormalize_bbox",unormalize_bbox)
    merge=draw_3d_bbox_wireframe_v2(shape,unormalize_bbox)
    print("merge",merge.shape)
    print("gt",gt.shape)
    save_nifti(merge,f"test_merge_{n_concat}.nii.gz")
    save_nifti(gt,f"test_gt_{n_concat}.nii.gz")

   
if __name__=="__main__":
    test_concat_bbox()