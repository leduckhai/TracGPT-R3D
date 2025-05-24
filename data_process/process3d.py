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

drop_keys=["Original","__index__level_0__","No.", "Column 9","Deliverable","Doctor","Start date","Google Drive Link","rotated_link","vn","fr","de","mandarin","korean","japanese","vi"]
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
        
        p_records={}
        patient_slide_to_bbox={}
        #  something like patient_slide_to_bbox["patient_id"][slide_id]=bbox
        for p_id in tqdm(p_ids):
            p_records[p_id]=defaultdict(list)
            patient_slide_to_bbox[p_id]={}
        for p_path in patient_json_files:
            
            with open(os.path.join(source_data,p_path), "rb") as f:
                data=json.load(f)
            for record in data:
                
                p_id=record["Patient ID"]
                for key,value in record.items():
                    if key not in drop_keys:
                        p_records[p_id][key].append(value)
                        #  we add to the list to remove duplicate data later on
                patient_slide_to_bbox[p_id][record["Slide"]]=record["A1"]
        
        # remove duplicate data
        for id,value in p_records.items():
            for column in p_records[id].keys():
                p_records[id][column]=list(set(p_records[id][column]))
                if len(p_records[id][column])==1:
                    p_records[id][column]=p_records[id][column][0]
                
            save_path=os.path.join(save_data_dir,f"{id}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(p_records[id], f, ensure_ascii=False, indent=2)
  
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
            print("all shapes",all_shapes)
            shape_counter=Counter(all_shapes)
            # reference_shape =  shape_counter.most_common(1)[0]
            reference_shape=None 
            max_count=0
            for shape,count in shape_counter.items():
                    if count>max_count:
                        reference_shape=shape
                        max_count=count
            keep_slides=[s for s in slide_base if slide_shape_map[s].shape==reference_shape]

        
            # mismatch_found = False
            # possible_bad_volume="bad_volume.txt"
            # if os.path.exists(possible_bad_volume):
            #     os.remove(possible_bad_volume)
            # for idx, slice_ in enumerate(slices_2d):
            #     if slice_.shape != reference_shape:
            #         print(f"❌ Mismatch at index {idx}: shape {slice_.shape} (expected {reference_shape})")
            #         if slice_.T.shape == reference_shape:
            #             print(f"🔁 Transposing slice at index {idx}")
            #             with open(possible_bad_volume, "a") as f:
            #                 f.write(f"{p_id} slice {idx}\n")
            #             slice_ = slice_.T
            #         else:
            #             mismatch_found = True
            #             break

            #     new_slices_2d.append(slice_)   

            # if not mismatch_found:
            #     print("✅ All slices have the same shape.")
                
            # new_slice_annot =[]
            # reference_shape = slices_annot_2d[0].shape
            # mismatch_found = False

            # for idx, slice_ in enumerate(slices_annot_2d):
            #     if slice_.shape != reference_shape:
            #         mismatch_found = True
            #         print(f"❌ Annot Mismatch at index {idx}: shape {slice_.shape} (expected {reference_shape})")
            #         if slice_.T.shape == reference_shape:
            #             print(f"🔁 Transposing slice at index {idx}")
            #             slice_ = slice_.T
            #             with open(possible_bad_volume,"a") as f:
            #                 f.write(f"{p_id} annot slice {idx}\n")
            #         else:
            #             mismatch_found = True
            #             break

            #     new_slice_annot.append(slice_)
            # if not mismatch_found:
            #     print("✅ All annot slices have the same shape.")

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
            


    print("target_root",target_root)
process_data()
