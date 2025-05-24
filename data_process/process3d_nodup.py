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

uid=str(uuid.uuid4())
num_concat=50
# this is the save directory
source_root="/home/ducnguyen/sync_local/repo/TracGPT/clean_data"
target_root="/home/ducnguyen/sync_local/repo/TracGPT/clean_data_3d"

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
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_annot_dir, exist_ok=True)
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_data_dir, exist_ok=True)
        os.makedirs(save_pick_dir, exist_ok=True)

        patient_json_files=os.listdir(source_data)
        p_ids=[path.split(".")[0] for path in patient_json_files]
        
      
  
        for p_id in tqdm(p_ids):
            
            img_slide_dir=os.path.join(source_img,p_id)
            annot_slide_dir=os.path.join(source_annot,p_id)

            slide_base= [f for f in os.listdir(img_slide_dir)]
            slide_base=sort_files(slide_base)

            
            
            slide_shape_map={}
            annot_shape_map={}
           
            print("process image pid",p_id)
            
            slices_2d=[]
            slices_annot_2d=[]
            for slide in slide_base:
                slide_path=os.path.join(img_slide_dir,slide)
                annot_path=os.path.join(annot_slide_dir,slide)
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
            # reference_shape =  shape_counter.most_common(1)[0]
            reference_shape=None 
            max_count=0
            for shape,count in shape_counter.items():
                    if count>max_count:
                        reference_shape=shape
                        max_count=count
            keep_slides=[s for s in slide_base if slide_shape_map[s].shape==reference_shape]

            #  save the index of keep_slides

            data_path=os.path.join(source_data,f"{p_id}.json")
            with open(data_path, "r") as f:
                data=json.load(f)

            slide_idx_map={}
            for i,slide in enumerate(slide_base):
                slide_idx_map[slide.split(".")[0]]=i
            # print("slide idx map",slide_idx_map)
            # print("data",data[0], "keep slides",keep_slides[0])
            data=[d for d in data if f"{d['Slide']}.pkl" in keep_slides]
            # return 
            for d in data:
                for k in drop_keys:
                    if k in d:
                        del d[k]
                d["slide_idx"]=slide_idx_map[d["Slide"]]
            with open(os.path.join(save_data_dir,f"{p_id}.json"), "w") as f:
                json.dump(data,f)


            image_3d=np.stack([slide_shape_map[s] for s in keep_slides], axis=0)
            annot_3d=np.stack([annot_shape_map[s] for s in keep_slides], axis=0)
            print("image_3d shape",image_3d.shape)
            print("annot_3d shape",annot_3d.shape)

            save_img_path=os.path.join(save_img_dir, f"{p_id}.pkl")
            save_annot_path=os.path.join(save_annot_dir, f"{p_id}.pkl")
            save_pick_slides_path=os.path.join(save_pick_dir, f"{p_id}.json")

            save_img_nifti_path=os.path.join(save_img_dir, f"{p_id}.nii.gz")
            save_annot_nifti_path=os.path.join(save_annot_dir, f"{p_id}.nii.gz")
            with open(save_img_path, "wb") as f:
                pickle.dump(image_3d, f)
            with open(save_annot_path, "wb") as f:
                pickle.dump(annot_3d, f)
            with open(save_pick_slides_path, "w") as f:
                json.dump(keep_slides, f)

            nifti_img = nib.Nifti1Image(image_3d, affine=np.eye(4))
            nib.save(nifti_img, save_img_nifti_path)

            nifti_img = nib.Nifti1Image(annot_3d, affine=np.eye(4))
            nib.save(nifti_img, save_annot_nifti_path)
        
        # p_records={}
        # patient_slide_to_bbox={}
        # #  something like patient_slide_to_bbox["patient_id"][slide_id]=bbox
        # for p_id in tqdm(p_ids):
        #     p_records[p_id]=defaultdict(list)
        #     patient_slide_to_bbox[p_id]={}

        # for p_path in patient_json_files:
            
        #     with open(os.path.join(source_data,p_path), "rb") as f:
        #         data=json.load(f)
        #     for record in data:
                
        #         p_id=record["Patient ID"]
        #         for key,value in record.items():
        #             if key not in drop_keys:
        #                 p_records[p_id][key].append(value)
        #                 #  we add to the list to remove duplicate data later on
        #         patient_slide_to_bbox[p_id][record["Slide"]]=record["A1"]
        
        # # remove duplicate data
        # for id,value in p_records.items():
        #     for column in p_records[id].keys():
        #         p_records[id][column]=list(set(p_records[id][column]))
        #         if len(p_records[id][column])==1:
        #             p_records[id][column]=p_records[id][column][0]
                
        #     save_path=os.path.join(save_data_dir,f"{id}.json")
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         json.dump(p_records[id], f, ensure_ascii=False, indent=2)
  



    print("target_root",target_root)
process_data()
