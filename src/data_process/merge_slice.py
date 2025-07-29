

from util import rgb_to_grayscale ,sort_files,combine_to_3d_bbox,draw_3d_bbox_wireframe,convert_list_slice_paths_to_3d,  merge_2d_bboxes_to_3d,group_and_merge_3d_bboxes,group_and_merge_3d_bboxes_v2,sort_files_v2
import os 
import json 
import pickle 
import nibabel as nib
from nibabel import Nifti1Image
import numpy as np 
import ast 
import shutil

def process():
    base_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train"
    p_id="OAS1_0002"
    data_file=os.path.join(base_dir,"data",f"{p_id}.json")  
    slice_dir=os.path.join(base_dir,"image",f"{p_id}")
    annot_dir=os.path.join(base_dir,"image_with_bboxes",f"{p_id}")

    n_concat=50
    slice_to_bbox={}

    slice_ls=os.listdir(slice_dir)
    with open(data_file, "rb") as f:
        data=json.load(f)

    for record in data:
        slide_id=record["Slide"]
        bbox=record["A1"]
        slice_to_bbox[slide_id]=bbox


    slice_basename=[path.split(".")[0] for path in slice_ls]
    # slices=sort_files(slice_basename)
    slices=sort_files_v2(slice_basename)
    
    print("slices",slices)
    # return
    slice_sample=slices[0]
    slice_sample_path=os.path.join(slice_dir,f"{slice_sample}.pkl")
    with open(slice_sample_path, "rb") as f:
        sample=pickle.load(f)
        print("data",sample.shape)
    height,width=sample.shape[0],sample.shape[1]


    output_dir="merged"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print("len slice",len(slices),"n concat",n_concat)
    for i in range(0,len(slices),n_concat):
        sapmle_slices=slices[i:i+n_concat]
        # print("len sample",len(sapmle_slices))
        slice_to_bboxes={}
        slice_pos={}
        
        # for j,slide_id in enumerate(sapmle_slices):
        #     bbox=ast.literal_eval(slice_to_bbox[slide_id].strip("'"))
        #     slice_to_bboxes[slide_id]=bbox
        #     slice_pos[slide_id]=j
        bounded_boxes=[ast.literal_eval(slice_to_bbox[slide_id].strip("'")) for slide_id in sapmle_slices]
        print("most bbox",len(bounded_boxes))
        unpacked_boxes=[]
        for k,boxes in enumerate(bounded_boxes):
            for j,bbox in enumerate(boxes):
                bbox.append(k)
                unpacked_boxes.append(bbox)
        slices_path=[os.path.join(slice_dir,f"{slice}.pkl") for slice in sapmle_slices]
        annot_path=[os.path.join(annot_dir,f"{slice}.pkl") for slice in sapmle_slices]
        
        bounded_boxeswith_depth=[]
        # bound_box_aggr=group_and_merge_3d_bboxes(slice_to_bboxes, slice_positions=slice_pos, img_size=  (width,height))
        bound_box_aggr=group_and_merge_3d_bboxes_v2(unpacked_boxes,  img_size=  (width,height),min_slices=2)

        print(f"sample {i} bbox",len(bound_box_aggr),bound_box_aggr)
        slice_3d=convert_list_slice_paths_to_3d(slices_path)
        print("slice 3d",slice_3d.shape)
        # merged_3d=draw_3d_bbox_wireframe(slice_3d,bound_box_aggr)

        annot_3d=convert_list_slice_paths_to_3d(annot_path)
        merge_file=os.path.join(output_dir,f"merge_{i}.nii.gz")
        slice_3d_file=os.path.join(output_dir,f"slice_{i}.nii.gz")
        annot_3d_file=os.path.join(output_dir,f"annot_{i}.nii.gz")
        # nifti_img = nib.Nifti1Image(merged_3d, affine=np.eye(4))
        # nib.save(nifti_img, merge_file)

        nifti_img = nib.Nifti1Image(slice_3d, affine=np.eye(4))
        nib.save(nifti_img, slice_3d_file)

        nifti_img = nib.Nifti1Image(annot_3d, affine=np.eye(4))
        nib.save(nifti_img, annot_3d_file)
        print("saved",merge_file,slice_3d_file,annot_3d_file)

if __name__ == "__main__":
    process()