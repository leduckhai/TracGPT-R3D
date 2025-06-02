import sys 
sys.path.append("/home/ducnguyen/sync_local/repo/TracGPT")
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
from data_process.util import group_and_merge_3d_bboxes_v2,convert_list_slice_paths_to_3d,draw_3d_bbox_wireframe_v2,draw_3d_bbox_filled,draw_3d_bbox_labels
import ast
drop_keys=["image","image_with_bboxes","Original","__index__level_0__","No.", "Column 9","Deliverable","Doctor","Start date","Google Drive Link","rotated_link","vn","fr","de","mandarin","korean","japanese","vi"]

dsc_path="data_process/desc.json"
with open(dsc_path,"r") as f:
    desc_map=json.load(f)

def test_concat_bbox():
    n_concat=50  
    # sample_file="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/data/OAS1_0079.json"
    sample_file="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/data/OAS1_0010.json"
    
    img_annot_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/image_with_bboxes"

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
        bbox_input.append(parsed)

    for i,bboxes in enumerate(bbox_input):
        print("bboxe idx",len(bboxes))
    output=group_and_merge_3d_bboxes_v2(bbox_input)
    print("output",len(output),output)
    
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
    print("len unormalize_bbox",len(unormalize_bbox))
    unormalize_bbox_input=[]
    for i,bboxes in enumerate(bbox_input):
        temp=[]
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min=x_min*shape[2]
            y_min=y_min*shape[1]
            z_min=i+1
            x_max=x_max*shape[2]
            y_max=y_max*shape[1]
            z_max=i+1
            temp.append([x_min, y_min, z_min, x_max, y_max, z_max])
        unormalize_bbox_input.extend(temp)
    bbox_v1=draw_3d_bbox_filled(shape,unormalize_bbox_input)
      
    # merge=draw_3d_bbox_wireframe_v2(shape,unormalize_bbox)
    merge=draw_3d_bbox_filled(shape,unormalize_bbox)
    labels=draw_3d_bbox_labels(shape,unormalize_bbox)
    print("merge",merge.shape)
    print("gt",gt.shape)
    save_nifti(merge,f"test_merge_{n_concat}.nii.gz")
    save_nifti(gt,f"test_gt_{n_concat}.nii.gz")
    save_nifti(bbox_v1,f"test_input_{n_concat}.nii.gz")
    save_nifti(labels,f"test_labels_{n_concat}.nii.gz")

   
if __name__=="__main__":
    test_concat_bbox()