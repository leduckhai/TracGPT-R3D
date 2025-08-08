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


dsc_path = "/root/TracGPT-R3D/src/data_process/desc.json"
with open(dsc_path, "r") as f:
    desc_map = json.load(f)

uid = str(uuid.uuid4())
num_concat = 32
tag = "overlap"

source_root = "/root/VLMTrac/2d_data"
target_root = "pseudo_3d"

target_root = os.path.join(target_root, f"{num_concat}_{tag}_slices", f"{uid}")
print("target root", target_root)
splits = ["train", "test"]


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


def process_data():
    for split in splits:
        source_dir = os.path.join(source_root, split)
        source_img = os.path.join(source_dir, "image")
        source_annot = os.path.join(source_dir, "image_with_bboxes")
        source_data = os.path.join(source_dir, "data")

        save_split_dir = os.path.join(target_root, split)
        save_data_dir = os.path.join(save_split_dir, "data")

        os.makedirs(save_data_dir, exist_ok=True)

        patient_json_files = os.listdir(source_data)
        p_ids = [path.split(".")[0] for path in patient_json_files]

        for p_id in tqdm(p_ids):
            with open(os.path.join(source_data, f"{p_id}.json"), "rb") as f:
                data = json.load(f)

            for d in data:
                for k in drop_keys:
                    if k in d:
                        del d[k]
                slide_data_map[d["Slide"]] = d
                
                
            os.makedirs(save_data_dir, exist_ok=True)

            img_slide_dir = os.path.join(source_img, p_id)
            annot_slide_dir = os.path.join(source_annot, p_id)

            slide_base = [f.split(".")[0] for f in os.listdir(img_slide_dir)]
            slide_subgroups = group_files(slide_base)
            print("group files",group_files)
            slide_shape_map = {}
            annot_shape_map = {}
            slide_data_map = {}


            patient_chunks = []
            for i, subgroup in enumerate(slide_subgroups):

                for slide in subgroup:
                    slide_path = os.path.join(img_slide_dir, f"{slide}.pkl")
                    annot_path = os.path.join(annot_slide_dir, f"{slide}.pkl")
                    with open(slide_path, "rb") as f:
                        slice_data = pickle.load(f)
                    with open(annot_path, "rb") as f:
                        annot_data = pickle.load(f)
                    slice_data = rgb_to_grayscale(slice_data)
                    annot_data = rgb_to_grayscale(annot_data)

                    slide_shape_map[slide] = slice_data
                    annot_shape_map[slide] = annot_data

                all_shapes = [slice_.shape for slice_ in slide_shape_map.values()]
                shape_counter = Counter(all_shapes)
                reference_shape = None
                max_count = 0
                for shape, count in shape_counter.items():
                    if count > max_count:
                        reference_shape = shape
                        max_count = count
                keep_slides = [
                    s for s in subgroup if slide_shape_map[s].shape == reference_shape
                ]

                if num_concat == -1:

                    chunk_slides = keep_slides
                    chunk_data = merge_slices(chunk_slides, slide_data_map)
                    patient_chunks.append(chunk_data)
                else:
                    for i in range(0, len(keep_slides), num_concat):
                        chunk_slides = keep_slides[i : i + num_concat]
                        if len(chunk_slides) < num_concat:
                            continue
                        chunk_data = merge_slices(chunk_slides, slide_data_map)
                        patient_chunks.append(chunk_data)

            print("call save")
            with open(os.path.join(save_data_dir, f"{p_id}.json"), "w") as f:
                json.dump(patient_chunks, f)
            return 
    print("target_root", target_root)


def merge_A3_data(list_A3):
    result = {"GCA": 0, "Koedam": 0, "MTA": 0}
    if not list_A3:
        return ""
    for s in list_A3:
        try:
            if s is not None:
                pairs = [pair.strip() for pair in s.split(",")]
                for pair in pairs:
                    key, value = pair.split("=")
                    result[key.strip()] = max(
                        result.get(key.strip(), 0), int(value.strip())
                    )
        except Exception as e:
            print("error merge_A3", s)
            print(e)
    print("result", result)
    return result


def merge_A4_data(list_A4):
    try:
        degree = [
            "Status (Non-Dementia)",
            "Status (Mild-Dementia)",
            "Status (Moderate-Dementia)",
        ]

        if not list_A4:  # Catches None, [], etc.
            print("Warning: Empty A4 list, defaulting to Non-Dementia")
            return "Non-Dementia"

        cleaned_list = [str(s) for s in list_A4 if s is not None]

        for level in reversed(degree):
            if any(fuzzy_contains(s.lower(), level.lower()) for s in cleaned_list):
                return level

        print("No dementia level found in A4, defaulting to Non-Dementia", cleaned_list)
        return "Non-Dementia"

    except Exception as e:
        print(f"Error processing A4: {str(e)}")
        print(f"Problematic A4 list: {list_A4}")
        return "Error A4"


def merge_A2_data(pairs):
    output = ""

    try:
        for key, value in pairs.items():
            output += desc_map[key][str(value)]
            output += "\n"
    except Exception as e:
        print("error in merge_A2", pairs)
        print(e)
    return output


def merge_A1_data(list_A1):
    bbox_input = []
    for bboxes_ls in list_A1:
        if bboxes_ls:
            try:
                parsed = ast.literal_eval(bboxes_ls)
            except Exception as e:
                print("error convert to bbox_A1")
                print(e)
                continue

            bbox_input.append(parsed)

    return bbox_input


def save_dsc_data(save_path="desc.json"):
    if os.path.exists(save_path):
        os.remove(save_path)
    data_dir = "/home/ubuntu/repo/TracGPT-R3D/clean_data/train/data"
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    result = {}

    for f in tqdm(files):
        with open(f, "r") as f:
            data = json.load(f)
        for d in data:
            note = d["Notes"]
            lines = [line.strip() for line in note.split("\n") if line.strip()]

            for line in lines:
                try:

                    scale_part, desc_part = line.split("=", 1)
                    scale = scale_part.strip()

                    score_part, description = desc_part.split(":", 1)
                    score = int(score_part.strip())
                    description = description.strip()

                    if scale not in result:
                        result[scale] = {}

                    result[scale][score] = description
                except Exception as e:
                    print(f"Error processing line: {line}")
                    print(f"Error message: {str(e)}")

    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)


def merge_slices(list_slices, slice_data_map):
    slice_data = [slice_data_map[s] for s in list_slices]
    output = {}

    output["Patient ID"] = slice_data[0]["Patient ID"]
    output["Q1"] = list(set([slice["Q1"] for slice in slice_data]))
    output["Q2"] = list(set([slice["Q2"] for slice in slice_data]))
    output["Q3"] = list(set([slice["Q3"] for slice in slice_data]))
    output["Q4"] = list(set([slice["Q4"] for slice in slice_data]))

    output["A1"] = merge_A1_data([slice["A1"] for slice in slice_data])
    output["A3"] = merge_A3_data([slice["A3"] for slice in slice_data])
    output["A2"] = merge_A2_data(output["A3"])
    output["A4"] = merge_A4_data([slice["A4"] for slice in slice_data])
    output["slice order"] = list_slices
    return output


if __name__ == "__main__":
    process_data()
