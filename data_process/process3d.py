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
from util import sort_files
import uuid
import nibabel as nib
import math
uid=str(uuid.uuid4())
num_concat=50
# this is the save directory
source_root="/home/ubuntu/repo/TracGPT-R3D/clean_data"
target_root="/home/ubuntu/repo/TracGPT-R3D/clean_data_3d"

target_root=os.path.join(target_root,f"{num_concat}_slices",f"{uid}")
print("target root",target_root)
splits=["train","test"]

def rgb_to_grayscale(img_rgb):
    """Convert (H, W, 3) RGB to (H, W) grayscale using standard weights."""
    return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]) 

drop_keys=["image","image_with_bboxes","Original","__index__level_0__","No.", "Column 9","Deliverable","Doctor","Start date","Google Drive Link","rotated_link","vn","fr","de","mandarin","korean","japanese","vi"]
def process_data():
    for split in splits:
        source_dir=os.path.join(source_root,split)
        source_img=os.path.join(source_dir,"image")
        source_annot=os.path.join(source_dir,"image_with_bboxes")
        source_data=os.path.join(source_dir,"data")
     
        save_split_dir=os.path.join(target_root,split)
        save_img_dir=os.path.join(save_split_dir,"image")
        save_annot_dir=os.path.join(save_split_dir,"image_with_bboxes")
        save_pick_dir=os.path.join(save_split_dir,"pick_slides")
        save_data_dir=os.path.join(save_split_dir,"data")

        patient_json_files=os.listdir(source_data)
        p_ids=[path.split(".")[0] for path in patient_json_files]
        
      
  
        for p_id in tqdm(p_ids):
            save_img_subdir=os.path.join(save_img_dir,p_id)
            save_annot_subdir=os.path.join(save_annot_dir,p_id)
            save_pick_subdir=os.path.join(save_pick_dir,p_id)

            os.makedirs(save_img_subdir, exist_ok=True),os.makedirs(save_img_dir, exist_ok=True)
            os.makedirs(save_annot_subdir, exist_ok=True)
            os.makedirs(save_pick_subdir, exist_ok=True)
            os.makedirs(save_data_dir, exist_ok=True)
            
            img_slide_dir=os.path.join(source_img,p_id)
            annot_slide_dir=os.path.join(source_annot,p_id)

            slide_base= [f.split(".")[0] for f in os.listdir(img_slide_dir)]
            slide_base=sort_files(slide_base)

            
            
            slide_shape_map={}
            annot_shape_map={}
           
            print("process image pid",p_id)
            
            slices_2d=[]
            slices_annot_2d=[]
            for slide in slide_base:
                slide_path=os.path.join(img_slide_dir,f"{slide}.pkl")
                annot_path=os.path.join(annot_slide_dir,f"{slide}.pkl")
                with open(slide_path, "rb") as f:
                    slice_data=pickle.load(f)
                with open(annot_path, "rb") as f:
                    annot_data=pickle.load(f)
                slice_data=rgb_to_grayscale(slice_data)
                annot_data=rgb_to_grayscale(annot_data)
                
                slide_shape_map[slide]=slice_data
                annot_shape_map[slide]=annot_data

            all_shapes=[slice_.shape for slice_ in slide_shape_map.values()]
            shape_counter=Counter(all_shapes)
            reference_shape=None 
            max_count=0
            for shape,count in shape_counter.items():
                    if count>max_count:
                        reference_shape=shape
                        max_count=count
            keep_slides=[s for s in slide_base if slide_shape_map[s].shape==reference_shape]

            slide_shape_map["background"]=np.zeros(reference_shape)
            annot_shape_map["background"]=np.zeros(reference_shape)

            slide_idx_map={}
            for i,slide in enumerate(keep_slides):
                slide_idx_map[slide]=i
                
            #  save the index of keep_slides

            
            for i in range(0,len(keep_slides),num_concat):
                chunk_idx=int(i/num_concat)
                chunk_slides=keep_slides[i:i+num_concat]
                if len(chunk_slides)<num_concat:
                    chunk_slides.extend(["background"]* (num_concat-len(chunk_slides)))
                assert len(chunk_slides)==num_concat 

                image_3d=np.stack([slide_shape_map[s] for s in chunk_slides], axis=0)
                annot_3d=np.stack([annot_shape_map[s] for s in chunk_slides], axis=0)

                save_img_path=os.path.join(save_img_subdir, f"{chunk_idx}.pkl")
                save_annot_path=os.path.join(save_annot_subdir, f"{chunk_idx}.pkl")
                # save_pick_slides_path=os.path.join(save_pick_dir, f"{i}.json")
                save_img_nifti_path=os.path.join(save_img_subdir, f"{chunk_idx}.nii.gz")
                save_annot_nifti_path=os.path.join(save_annot_subdir, f"{chunk_idx}.nii.gz")
                with open(save_img_path, "wb") as f:
                    pickle.dump(image_3d, f)
                # with open(save_annot_path, "wb") as f:
                #     pickle.dump(annot_3d, f)
                # with open(save_pick_slides_path, "w") as f:
                    # json.dump(keep_slides, f)

                # nifti_img = nib.Nifti1Image(image_3d, affine=np.eye(4))
                # nib.save(nifti_img, save_img_nifti_path)

                # nifti_img = nib.Nifti1Image(annot_3d, affine=np.eye(4))
                # nib.save(nifti_img, save_annot_nifti_path)

            data_path=os.path.join(source_data,f"{p_id}.json")
            with open(data_path, "r") as f:
                data=json.load(f)

            data=[d for d in data if d['Slide'] in keep_slides]

            
            # drop no need keys
            for d in data:
                for k in drop_keys:
                    if k in d:
                        del d[k]
                
                d["ori_slide_idx"]=slide_idx_map[d["Slide"]]
                d["chunk_idx"]=int(slide_idx_map[d["Slide"]]/ num_concat)
                d["slide_in_chunk_th"]=slide_idx_map[d["Slide"]] % num_concat

            with open(os.path.join(save_data_dir,f"{p_id}.json"), "w") as f:
                json.dump(data,f)

    print("target_root",target_root)
process_data()
