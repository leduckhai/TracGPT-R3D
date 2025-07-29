import os
import json 
from util import convert_list_slice_paths_to_3d
import numpy as np
import nibabel as nib

base_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data_v2/4a03e20b-94ef-431c-890c-444bbf89d30b/train"

reduce_base="reduce"
os.makedirs(reduce_base,exist_ok=True)

prop_paths=[os.path.join(base_dir,f) for f in os.listdir(base_dir) if f.startswith("prop_") and f.endswith(".json")]
sorted(prop_paths)
base_slides="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/image/OAS1_0003"
base_annots="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/image_with_bboxes/OAS1_0003"
def process():
        p="/home/ducnguyen/sync_local/repo/TracGPT/clean_data_v2/4a03e20b-94ef-431c-890c-444bbf89d30b/train/prop_0.json"
        # for p in prop_paths:
        with open(p,"r") as f:
            data=json.load(f)

        data_map=[]
        print("prop",p, "len data",len(data))
        for record in data:
        
            data_map.append((record["Deliverable"],record["A1"],record["Slide"]))
        data_map=sorted(data_map,key=lambda x:x[0])
       
        deliverables=[d[0] for d in data_map]
        print("deliverables",deliverables)
        slides=[d[2] for d in data_map]
        print("slides",slides)
        slide_paths=[os.path.join(base_slides,f"{s}.pkl") for s in slides]
        annot_paths=[os.path.join(base_annots,f"{s}.pkl") for s in slides]
        img_3d=convert_list_slice_paths_to_3d(slide_paths)
        annot_3d=convert_list_slice_paths_to_3d(annot_paths)
        print("img_3d shape",img_3d.shape)
        print("annot_3d shape",annot_3d.shape)
        affine = np.eye(4)

        nifti_img = nib.Nifti1Image(img_3d, affine)
        if os.path.exists(f"file_concat.nii"):
            os.remove(f"file_concat.nii")
        nib.save(nifti_img, f"file_concat.nii")

        nifti_img = nib.Nifti1Image(annot_3d, affine)
        if os.path.exists(f"file_concat_annot.nii"):
            os.remove(f"file_concat_annot.nii")
        nib.save(nifti_img, f"file_concat_annot.nii")
        return
if __name__=="__main__": 
    process()