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
class TracDataset2D(Dataset):
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

if __name__ == "__main__":
    data_paths = [
        "/root/VLMTrac/chunks/train/data/1.json",
        "/root/VLMTrac/chunks/train/data/2.json",
    ]
    image_path = "/root/VLMTrac/2d_data/train/image"
    dataset = TracDatasetWhite(data_paths, image_path)
    print(dataset[0])