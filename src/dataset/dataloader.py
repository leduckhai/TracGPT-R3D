import torch
import os
import sys
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
from src.data_process.util import convert_list_slice_paths_to_3d
import os
import numpy as np
import json
from src.dataset.trac_dataset import TracDataset
from src.dataset.trac_dataset_white import TracDatasetWhite
from src.dataset.trac_dataset_cot import TracDatasetCoT

def load_data(
    train_val_dir="/root/VLMTrac/chunks/train/data",
    test_dir="/root/VLMTrac/chunks/train/data",
    image_train_path="/root/VLMTrac/2d_data/train/image",
    image_test_path="/root/VLMTrac/2d_data/test/image",
    dataset="trac",
    train_sample=-1,
    val_sample=-1,
    test_sample=-1,
    original=False,
    overfit_train=False,
):
    if dataset == "trac":
        dataset = TracDataset
    elif dataset == "trac_white":
        dataset = TracDatasetWhite
    elif dataset == "trac_cot":
        dataset = TracDatasetCoT
    train_data_paths = [
        os.path.join(train_val_dir, record) for record in os.listdir(train_val_dir)
    ]
    test_data_paths = [
        os.path.join(test_dir, record) for record in os.listdir(test_dir)
    ]
    train_paths, val_paths = train_test_split(
        train_data_paths, test_size=0.1, random_state=12
    )
    train_mode="train"
    val_mode="val"
    test_mode="test"                
    if overfit_train:
        print("Overfitting on training set")
        test_data_paths = train_paths
        image_test_path = image_train_path
        val_paths = train_paths[:len(val_paths)]
        # we set val mode =test to leave the answer empty string for generation
        val_mode="test"
        
    train_set = dataset(
        data_paths=train_paths,
        image_path=image_train_path,
        mode=train_mode ,
        n_sample=train_sample,
    )
    val_set = dataset(
        data_paths=val_paths,
        image_path=image_train_path,
        mode=val_mode,
        n_sample=val_sample,
    )
    test_set = dataset(
        data_paths=test_data_paths,
        image_path=image_test_path,
        mode=test_mode,
        n_sample=test_sample,
    )
    return train_set, val_set, test_set

from src.data_process.util import save_nifti

def save_3d_data(data_paths, image_dir, save_path):
    p_id=set()
    process_pid= set()
    for path in data_paths:
        with open(path, "r") as f:
            data = json.load(f)
        sample=data[0]
        patient_id = sample["Patient ID"]
        slice_order = sample["slice order"]
        p_id.add(patient_id)
        if patient_id in process_pid:
            print("Patient ID already processed:", patient_id)
        
        image_paths = [os.path.join(image_dir, patient_id, f"{s}.pkl") for s in slice_order]
        img_3d=convert_list_slice_paths_to_3d(image_paths)

        save_file_path= os.path.join(save_path, f"{patient_id}.nii.gz")
        save_nifti(img_3d, save_file_path)
        process_pid.add(patient_id)
        print(f"Saved 3D data for patient {patient_id} to {save_path}")
    print("Total unique patient IDs processed:", len(p_id))
        
if __name__ == "__main__":
    import os
    
    train_val_dir = "pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/train/data"
    test_dir = "pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/test/data"
    image_train_path = "viz_data/train/image"
    image_test_path = "viz_data/test/image"
    # os.makedirs(image_train_path, exist_ok=True)
    # os.makedirs(image_test_path, exist_ok=True)
    # save_3d_data(
    #     data_paths=[os.path.join(train_val_dir, f) for f in os.listdir(train_val_dir) if f.endswith('.json')],
    #     image_dir=image_train_path,
    #     save_path="clean_data/train/3d_data",
    # )
    # save_3d_data(
    #     data_paths=[os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')],
    #     image_dir=image_test_path,
    #     save_path="clean_data/test/3d_data",
    # )
    # train_set, val_set, test_set = load_data(
    #     train_val_dir="pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/train/data",
    #     test_dir="pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/test/data",
    #     image_train_path="clean_data/train/image",
    #     image_test_path="clean_data/test/image",
    #     dataset="trac_white",
    # )
    # # print("len trainset", len(train_set), len(val_set), len(test_set))
    # answer_set=set()
    # for i, sample in enumerate(train_set):
    #     if i == 3:
    #         break
    #     # print(sample.keys())
    #     Q1 = sample["Q1"]
    #     A1 = sample["A1"]
    #     Q2 = sample["Q2"]
    #     A2 = sample["A2"]
    #     Q3 = sample["Q3"]
    #     A3 = sample["A3"]
    #     Q4 = sample["Q4"]
    #     A4 = sample["A4"]
        
    #     answer_set.add(A4)
    # print("answer set", answer_set)
        # # print("slice order", slice_order)
        # print("Q1", Q1)
        # # Q1: bbox
    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    import yaml
    from transformers import AutoTokenizer
    import torch

    config_path="/root/TracGPT-R3D/config/vit_llama_3B.yaml"
    # config_path = "/workspace/TracGPT-R3D/config/vit_llama_3B_80GB.yaml"
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    custom_config = full_config["model"]["config"]
    base_model_name = custom_config["language_model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model_name = custom_config["language_model"]["name"]
    print("Loading base model:", base_model_name)

    new_tokens = ["<image>", "<PAD>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    tokenizer.pad_token = "<PAD>"
    print("pad token id:", tokenizer.pad_token_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nAfter modification:")
    print("pad_token:", tokenizer.pad_token)
    print("padding_side:", tokenizer.padding_side)
    print("vocab_size:", len(tokenizer))
    print("eos_token_id:", tokenizer.eos_token_id)
    data_config = full_config["data"]
    train_set, val_set, test_set = load_data(
        train_val_dir=data_config["train_val_dir"],
        test_dir=data_config["test_dir"],
        image_train_path=data_config["image_train_path"],
        image_test_path=data_config["image_test_path"],
        dataset=data_config["dataset"],
        train_sample=data_config["train_sample"],
        val_sample=data_config["val_sample"],
        test_sample=data_config["test_sample"],
        dataset_config=data_config["dataset_config"],
    )
    
    for i in range(len(train_set)):
        sample = train_set[i]
        label=sample["answer"]
        print("train label", label)
        
        # if i >= 20:
        #     break

    # collator = load_collator(full_config["general"]["collator"], tokenizer=tokenizer)
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=2,
    #     shuffle=True,
    #     collate_fn=collator,
    #     num_workers=0,
    #     pin_memory=True,
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_set,
    #     batch_size=2,
    #     shuffle=False,
    #     collate_fn=collator,
    #     num_workers=0,
    #     pin_memory=True,
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_set,
    #     batch_size=2,
    #     shuffle=False,
    #     collate_fn=collator,
    #     num_workers=0,
    #     pin_memory=True,
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=2,
    #     shuffle=False,
    #     collate_fn=collator,
    #     num_workers=0,
    #     pin_memory=True,
    # )
    # for i, batch in enumerate(train_loader):
    #     print("train loader")
    #     images = batch["images"]
    #     print("image load shape", images.shape)
    #     # print("A1", A1)   
    #     if i >= 3:
    #         break
    # for i, batch in enumerate(val_loader):
    #     print("val loader")
    #     images = batch["images"]
    #     # status_criteria = batch["status_criteria"]
    #     print("images  load shape", images.shape)
    #     if i >= 3:
    #         break
       
    #     # print("status_criteria", status_criteria)
    # for i, batch in enumerate(test_loader):
    #     print("test loader")
    #     images = batch["images"]
      
    #     # status_criteria = batch["status_criteria"]
    #     print("images load shape", images.shape)
    #     if i >= 3:
    #         break
       
        # print("status_criteria", status_criteria)
    # for i, sample in enumerate(test_set):
    #     # if i == 3:
    #     #     break
    #     print(sample.keys())
    #     slice_order = sample["slice_order"]
    #     patient_id = sample["Patient_ID"]
    #     Q1 = sample["Q1"]
    #     A1 = sample["A1"]
    #     Q2 = sample["Q2"]
    #     A2 = sample["A2"]
    #     Q3 = sample["Q3"]
    #     A3 = sample["A3"]
    #     Q4 = sample["Q4"]
    #     A4 = sample["A4"]
    #     bbox= sample["bbox"]
    #     print("patient id", patient_id)
    #     # print("slice order", slice_order)
    #     print("Q1", Q1)
    #     # Q1: bbox

    #     print("A1", A1)
    #     print("len A1", len(A1))
    #     # print("Q2", Q2)
    #     # print("A2", A2)
    #     # print("Q3", Q3)
    #     # print("A3", A3)
    #     # print("Q4", Q4)
    #     # Q4 mild dementia,...
    #     # print("A4", A4)
    #     image = sample["image"]
    #     print("image shape", image.shape, image.min(), image.max())
        # image = np.array(image.squeeze(0))
        # print("image shape", image.shape, image.min(), image.max())
