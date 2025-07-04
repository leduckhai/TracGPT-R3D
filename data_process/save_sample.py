import json 
import os
import pickle
import nibabel as nib
import shutil

from util import draw_3d_bbox_filled,draw_3d_bbox_wireframe,convert_list_slice_paths_to_3d,save_nifti
def process():
    sample_file="/home/ducnguyen/sync_local/repo/TracGPT/clean_data_3d/-1_overlap_slices/175c930c-edb9-4e33-bcd0-6d091228f704/train/data/OAS1_0056.json"
    img_annot_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/image_with_bboxes"
    with open(sample_file,"r") as f:
        data=json.load(f)
    p_id=data[0]["Patient ID"]
    for i,sample in enumerate(data):
        slices=sample["slice order"]
        n_concat=len(slices)
        all_annot_paths=[os.path.join(img_annot_dir,p_id,f"{s}.pkl") for s in slices]
        grount_truth=convert_list_slice_paths_to_3d(all_annot_paths)
        with open(all_annot_paths[0],"rb") as f:
            slide=pickle.load(f)
        bboxes=sample["A1"]
        
        shape=(n_concat,slide.shape[0],slide.shape[1])

        unormalize_bbox=[]
        for bbox in bboxes:
            x_min, y_min, z_min, x_max, y_max, z_max = bbox
            print("bbox",bbox)
            x_min=x_min*shape[2]
            y_min=y_min*shape[1]
            z_min=z_min*shape[0]
            x_max=x_max*shape[2]
            y_max=y_max*shape[1]
            z_max=z_max*shape[0]
            unormalize_bbox.append([x_min, y_min, z_min, x_max, y_max, z_max])
        print("len unormalize_bbox",len(unormalize_bbox))
        bboxes_colors=draw_3d_bbox_filled(shape,unormalize_bbox)
        # merge=draw_3d_bbox_wireframe_v2(shape,unormalize_bbox)
        # merge=draw_3d_bbox_filled(shape,unormalize_bbox)
        # labels=draw_3d_bbox_labels(shape,unormalize_bbox)
        
        pred_path=f"test_pred_{i}_{n_concat}.nii.gz"
        if os.path.exists(pred_path):
            os.remove(pred_path)
        gt_path=f"test_gt_{i}_{n_concat}.nii.gz"
        if os.path.exists(gt_path):
            os.remove(gt_path)

      
        save_nifti(grount_truth,f"test_gt_{i}_{n_concat}.nii.gz")
        save_nifti(bboxes_colors,f"test_pred_{i}_{n_concat}.nii.gz")


    

if __name__ == "__main__":
    process()