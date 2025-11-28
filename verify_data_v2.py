import os
import json
from tqdm import tqdm
import shutil
from collections import defaultdict
from src.dataset.dataloader import load_data
from src.data_process.util import bboxes_to_filled_volume, bboxes_to_wireframe_volume,draw_3d_bbox_wireframe_v2, draw_3d_bbox_filled,draw_3d_bbox_wireframe,convert_list_slice_paths_to_3d,save_nifti,group_and_merge_3d_bboxes_v2

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("leduckhai/S-Chain", "English")

def save_data(dataset, save_image_dir, save_labels_dir):
    if os.path.exists(save_image_dir):
        shutil.rmtree(save_image_dir)
    if os.path.exists(save_labels_dir):
        os.remove(save_labels_dir)
    os.makedirs(save_image_dir, exist_ok=True)
    save_label_dir="verify_label"
    os.makedirs(save_label_dir, exist_ok=True)
    test_labels=[]

    for i, sample in enumerate(tqdm(dataset, desc="Testing")):
        p_id,Q1,A1,Q2,A2,Q3,A3,Q4,A4=sample["P_ID"],sample["Q1"],sample["A1"],sample["Q2"],sample["A2"],sample["Q3"],sample["A3"],sample["Q4"],sample["A4"]
        # bboxes=group_and_merge_3d_bboxes_v2(A1, num_concat=32)
        image=sample["image"]
        # print("iamge shape:", image.shape)
        save_image_path=os.path.join(save_image_dir,f"image_{i}.nii.gz")
        save_label_path=os.path.join(save_label_dir,f"bbox_{i}.nii.gz")
        save_nifti(image, save_image_path)
        # save_nifti(bboxes_to_filled_volume(image.shape,bboxes), save_label_path)
        meta={}
        meta["sample_index"]=i
        meta["A4"]=A4
        meta["P_ID"]=p_id
        test_labels.append(meta)
        # break
        
    
    with open(save_labels_dir, "w") as f:
        json.dump(test_labels, f, indent=4)
    print(f"Saved test labels to {save_labels_dir}")
if __name__ == "__main__":
    import numpy as np
    from src.data_process.util import convert_list_slice_paths_to_3d,convert_list_slice_paths_with_bbox_to_3d
    train_val_dir= "pseudo_3d/-1_all_slices/7c75e990-60b0-44d4-82f4-45ec2513e97e/train/data"
    # pseudo_3d/-1_all_slices/7c75e990-60b0-44d4-82f4-45ec2513e97e/train/data": s_change full
    test_dir="pseudo_3d/-1_all_slices/7c75e990-60b0-44d4-82f4-45ec2513e97e/test/data"
    image_train_path="clean_data_s_chain/train/image"
    image_test_path= "clean_data_s_chain/test/image"
    
    # train_val_dir= "psuteudo_3d/-1_all_slices/07eaa1bc-b53a-4126-accf-92509623ecdb/train/data"
    # test_dir="pseudo_3d/-1_all_slices/07eaa1bc-b53a-4126-accf-92509623ecdb/test/data"
    # image_train_path="clean_data/train/image_with_bboxes"
    # image_test_path= "clean_data/test/image_with_bboxes"
    train_set, val_set, test_set = load_data(
        train_val_dir=train_val_dir,
        test_dir=test_dir,
        image_train_path=image_train_path,
        image_test_path=image_test_path,
        dataset="trac_verify",
        original=True,
    )   
    print("len train_set:", len(train_set))
    print("len val_set:", len(val_set))
    print("len test_set:", len(test_set))

    save_data(train_set, "verify_image/train", "verify_label/train_labels.json")
    save_data(val_set, "verify_image/val", "verify_label/val_labels.json")
    save_data(test_set, "verify_image/test", "verify_label/test_labels.json")
    
 
    image=np.random.rand(2,256,256)
    # image_path=["clean_data_s_chain/train/image/OAS1_0002/mpr-1_101.pkl"]
    # boxes=[[[0.2,0.2,0.4,0.4],[0.5,0.5,0.8,0.8]]]
    # image=convert_list_slice_paths_with_bbox_to_3d(image_path,boxes)
    