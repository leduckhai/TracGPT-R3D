import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import pickle
import sys
import os
from dotenv import load_dotenv
import random
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

from src.data_process.save_sample import process_sample


class TracDatasetWhite(Dataset):
    def __init__(
        self,
        data_paths,
        image_path,
        mode="train",
        n_sample=-1,
        image_shape=[32, 256, 256],
        bbox_only=False,
    ):
        self.image_shape = image_shape

        self.mode = mode
        self.base_transform = Compose(
            [
                EnsureChannelFirstD(keys=["image"], channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=255,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ResizeD(
                    keys=["image"],
                    spatial_size=[32, 256, 256],
                    mode="trilinear",
                    size_mode="all",
                ),
            ]
        )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensord(keys=["image"], dtype=torch.float),
            ]
        )
        self.img_dir = image_path

        self.qa_banks = []
        qa_maps = {
            "Q1": "A1",
            "Q2": "A2",
            "Q3": "A3",
            "Q4": "A4",
        }
        print("data path",data_paths)
        for path in data_paths:
            
            with open(path, "r") as f:
                data = json.load(f)

            for sample in data:
                data_point = {
                    "slice_order": sample["slice order"],
                    "Patient_ID": sample["Patient ID"],
                    "Q1": sample["Q1"],
                    "Q2": sample["Q2"],
                    "Q3": sample["Q3"],
                    "Q4": sample["Q4"],
                    "A1": sample["A1"],
                    "A2": sample["A2"],
                    "A3": sample["A3"],
                    "A4": sample["A4"],
                }

                self.qa_banks.append(data_point)
        if bbox_only:
            self.qa_banks = [d for d in self.qa_banks if d["answer_type"] == "bbox_3d"]
        if n_sample != -1:
            self.qa_banks = self.qa_banks[:n_sample]
        random.shuffle(self.qa_banks)

    def __len__(self):
        return len(self.qa_banks)

    def __getitem__(self, idx):
        data_point = self.qa_banks[idx]
        slice_order = data_point["slice_order"]
        patient_id = data_point["Patient_ID"]

        image_path = [
            os.path.join(self.img_dir, patient_id, f"{s}.pkl") for s in slice_order
        ]
        for path in image_path:
            assert os.path.exists(path), f"{path} does not exist"
        image_3d = convert_list_slice_paths_to_3d(image_path)
        image_dict = self.base_transform({"image": image_3d})

        data_point["image"] = image_dict["image"]
        return data_point


class TracDataset(Dataset):
    def __init__(
        self,
        data_paths,
        image_path,
        mode="train",
        n_sample=-1,
        image_shape=[32, 256, 256],
        bbox_only=False,
    ):
        self.image_shape = image_shape

        self.mode = mode
        self.base_transform = Compose(
            [
                EnsureChannelFirstD(keys=["image"], channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=255,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ResizeD(
                    keys=["image"],
                    spatial_size=[32, 256, 256],
                    mode="trilinear",
                    size_mode="all",
                ),
            ]
        )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensord(keys=["image"], dtype=torch.float),
            ]
        )
        self.img_dir = image_path

        self.qa_banks = []
        qa_maps = {
            "Q1": "A1",
            "Q2": "A2",
            "Q3": "A3",
            "Q4": "A4",
        }
        for path in data_paths:
            with open(path, "r") as f:
                data = json.load(f)

            for sample in data:
                for q, a in qa_maps.items():
                    data_point = {
                        "slice_order": sample["slice order"],
                        "Patient_ID": sample["Patient ID"],
                        "question": sample[q],
                        "answer": sample[a],
                    }
                    if q == "Q1":
                        data_point["answer_type"] = "bbox_3d"
                        data_point["bbox_3d"] = sample[a]
                    else:
                        data_point["answer_type"] = "text"
                        data_point["bbox_3d"] = None
                    self.qa_banks.append(data_point)
        if bbox_only:
            self.qa_banks = [d for d in self.qa_banks if d["answer_type"] == "bbox_3d"]
        if n_sample != -1:
            self.qa_banks = self.qa_banks[:n_sample]
        random.shuffle(self.qa_banks)

    def __len__(self):
        return len(self.qa_banks)

    def __getitem__(self, idx):
        data_point = self.qa_banks[idx]
        slice_order = data_point["slice_order"]
        patient_id = data_point["Patient_ID"]

        image_path = [
            os.path.join(self.img_dir, patient_id, f"{s}.pkl") for s in slice_order
        ]
        for path in image_path:
            assert os.path.exists(path), f"{path} does not exist"
        image_3d = convert_list_slice_paths_to_3d(image_path)
        image_dict = self.base_transform({"image": image_3d})

        data_point["image"] = image_dict["image"]
        return data_point


def load_data(
    train_val_dir="/root/VLMTrac/chunks/train/data",
    image_path="/root/VLMTrac/2d_data/train/image",
    bbox_only=False,
    dataset="trac",
    train_sample=1000,
    val_sample=500,
    test_sample=100,
):
    if dataset == "trac":
        dataset = TracDataset
    elif dataset == "trac_white":
        dataset = TracDatasetWhite

    data_paths = [
        os.path.join(train_val_dir, record) for record in os.listdir(train_val_dir)
    ]
    print("data paths", len(data_paths))
    train_paths, test_paths = train_test_split(
        data_paths, test_size=0.2, random_state=42
    )
    train_paths, val_paths = train_test_split(
        train_paths, test_size=0.2, random_state=42
    )
    train_set = dataset(
        data_paths=train_paths,
        image_path=image_path,
        mode="train",
        n_sample=train_sample,
        bbox_only=bbox_only,
    )
    val_set = dataset(
        data_paths=val_paths, image_path=image_path, mode="val", n_sample=val_sample
    )
    test_set = dataset(
        data_paths=test_paths, image_path=image_path, mode="test", n_sample=test_sample
    )
    return train_set, val_set, test_set


if __name__ == "__main__":
    import os

    # train_set, val_set, test_set = load_data()
    train_set, val_set, test_set = load_data(
        train_val_dir="/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/0691cd9f-8dad-4005-811d-34fb610d4f88/train/data",
        dataset="trac_white",
    )
    for i, sample in enumerate(train_set):
        if i == 3:
            break
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
        print("A1", A1)
        print("Q2", Q2)
        print("A2", A2)
        print("Q3", Q3)
        print("A3", A3)
        print("Q4", Q4)
        print("A4", A4)
        image = sample["image"]
        image = np.array(image.squeeze(0))
        print("image shape", image.shape, image.min(), image.max())
