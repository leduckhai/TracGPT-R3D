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
import monai.transforms as mtf
import os
import numpy as np
import json
from monai.transforms import Compose, ResizeD, EnsureChannelFirstD, SqueezeDimD
from monai.transforms import ScaleIntensityRanged
from src.dataset.trac_dataset import TracDataset
from src.dataset.trac_dataset_white import TracDatasetWhite


def load_data(
    train_val_dir="/root/VLMTrac/chunks/train/data",
    test_dir="/root/VLMTrac/chunks/train/data",
    image_train_path="/root/VLMTrac/2d_data/train/image",
    image_test_path="/root/VLMTrac/2d_data/test/image",
    dataset="trac",
    train_sample=-1,
    val_sample=-1,
    test_sample=-1,
    dataset_config: dict = {},
):
    if dataset == "trac":
        dataset = TracDataset
    elif dataset == "trac_white":
        dataset = TracDatasetWhite

    train_data_paths = [
        os.path.join(train_val_dir, record) for record in os.listdir(train_val_dir)
    ]
    test_data_paths = [
        os.path.join(test_dir, record) for record in os.listdir(test_dir)
    ]

    train_paths, val_paths = train_test_split(
        train_data_paths, test_size=0.1, random_state=42
    )
    train_set = dataset(
        data_paths=train_paths,
        image_path=image_train_path,
        mode="train",
        n_sample=train_sample,
        dataset_config=dataset_config,
    )
    val_set = dataset(
        data_paths=val_paths,
        image_path=image_train_path,
        mode="val",
        n_sample=val_sample,
        dataset_config=dataset_config,
    )
    test_set = dataset(
        data_paths=test_data_paths,
        image_path=image_test_path,
        mode="test",
        n_sample=test_sample,
        dataset_config=dataset_config,
    )
    return train_set, val_set, test_set


if __name__ == "__main__":
    import os

    # train_set, val_set, test_set = load_data()
    train_set, val_set, test_set = load_data(
        train_val_dir="/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/1ad40bdc-da39-4dd8-9d3a-27ce00e754fa/train/data",
        test_dir="/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/1ad40bdc-da39-4dd8-9d3a-27ce00e754fa/test/data",
        dataset="trac_white",
    )
    print("len trainset", len(train_set), len(val_set), len(test_set))
    for i, sample in enumerate(test_set):
        if i == 3:
            break
        print(sample.keys())
        slice_order = sample["slice_order"]
        patient_id = sample["Patient_ID"]
        Q1 = sample["Q1"]
        A1 = sample["A1"]
        Q2 = sample["Q2"]
        A2 = sample["A2"]
        Q3 = sample["Q3"]
        A3 = sample["A3"]
        Q4 = sample["Q4"]
        A4 = sample["A4"]
        print("patient id", patient_id)
        print("slice order", slice_order)
        print("Q1", Q1)
        # Q1: bbox

        print("A1", A1)
        print("len A1", len(A1))
        # print("Q2", Q2)
        # print("A2", A2)
        # print("Q3", Q3)
        # print("A3", A3)
        # print("Q4", Q4)
        # Q4 mild dementia,...
        # print("A4", A4)
        image = sample["image"]
        print("image shape", image.shape, image.min(), image.max())
        # image = np.array(image.squeeze(0))
        # print("image shape", image.shape, image.min(), image.max())
