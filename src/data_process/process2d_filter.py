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
import math
from util import group_and_merge_3d_bboxes_v2, group_files
import ast
from difflib import SequenceMatcher


def fuzzy_contains(s, d, threshold=0.9):
    """
    Checks if a substring `d` approximately exists in `s` with a given similarity threshold.
    Returns True if the best match has a similarity ratio >= threshold.
    """
    len_d = len(d)
    if len_d == 0:
        return True  # Empty string always matches

    best_ratio = 0
    for i in range(len(s) - len_d + 1):
        window = s[i : i + len_d]
        ratio = SequenceMatcher(None, window, d).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
        if best_ratio >= threshold:
            return True
    return best_ratio >= threshold




def rgb_to_grayscale(img_rgb):
    """Convert (H, W, 3) RGB to (H, W) grayscale using standard weights."""
    return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])


drop_keys = [
    "image",
    "image_with_bboxes",
    "Original",
    "__index__level_0__",
    "No.",
    "Column 9",
    "Deliverable",
    "Doctor",
    "Start date",
    "Google Drive Link",
    "rotated_link",
    "vn",
    "fr",
    "de",
    "mandarin",
    "korean",
    "japanese",
    "vi",
]


class Process2DFilter:
    def __init__(self, num_concat = 32, tag = "no_overlap_2d", source_root = "/root/VLMTrac/2d_data", target_root = "only_2d", splits = ["train", "test"], dsc_path = "/root/TracGPT-R3D/src/data_process/desc.json"):
        self.num_concat = num_concat
        self.tag = tag
        self.source_root = source_root
        self.target_root = target_root
        self.splits = splits
        self.dsc_path = dsc_path
        self.uid = str(uuid.uuid4())
        self.target_root = os.path.join(self.target_root, f"{self.num_concat}_{self.tag}_slices", f"{self.uid}")
        print("target root", self.target_root)
    def process_data(self):
        with open(self.dsc_path, "r") as f:
            desc_map = json.load(f)

        for split in self.splits:
            source_dir = os.path.join(self.source_root, split)
            source_img = os.path.join(source_dir, "image")
            source_annot = os.path.join(source_dir, "image_with_bboxes")
            source_data = os.path.join(source_dir, "data")

            save_split_dir = os.path.join(self.target_root, split)
            save_data_dir = os.path.join(save_split_dir, "data")

            os.makedirs(save_data_dir, exist_ok=True)

            patient_json_files = os.listdir(source_data)
            p_ids = [path.split(".")[0] for path in patient_json_files]
            print("Total p_ids",len(p_ids))
            total_data=0
            for i, p_id in enumerate(tqdm(p_ids)):
                with open(os.path.join(source_data, f"{p_id}.json"), "rb") as f:
                    data = json.load(f)
                print(f"Len data for {p_id}",len(data))
                total_data+=len(data)

                slide_data_map = {}
                for d in data:
                    for k in drop_keys:
                        if k in d:
                            del d[k]
                    slide_data_map[d["Slide"]] = d
                    
                with open(os.path.join(save_data_dir, f"{p_id}.json"), "w") as f:
                    json.dump(data, f, indent=4)
                    # json.dump(slide_data_map, f, indent=4)
         
        print("total data",total_data)
        print("target_root", self.target_root)




if __name__ == "__main__":
    # process_data()
    data_process = Process2DFilter()
    data_process.process_data()
