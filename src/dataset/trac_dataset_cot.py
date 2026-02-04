import monai
from monai.transforms import  Compose
from monai.transforms import Compose, ResizeD, EnsureChannelFirstD,ScaleIntensityRanged
original_set_random_state = monai.transforms.compose.Compose.set_random_state
from torch.utils.data import Dataset
import os
import json
import sys
from dotenv import load_dotenv
import random
load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
from src.data_process.util import convert_list_slice_paths_to_3d
from collections import defaultdict
import pandas as pd
import ast 

def get_base_transform(spatial_size=[32, 256, 256]):
    return Compose(
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
                    spatial_size=spatial_size,
                    mode="trilinear",
                    size_mode="all",
                ),
            ]
        )

class TracDatasetCoT(Dataset):
    def __init__(self, data_paths, image_path, mode="train", n_sample=-1):
        random.seed(68)  
        self.img_dir = image_path
        self.sample_indices = []    
        self.mode = mode
        self.height = 256
        self.width = 256
        significant_slice_path="significant_slice.csv"
        df = pd.read_csv(significant_slice_path)
        self.significant_slices_data = (
        df.dropna(subset=["Slide"])       
        .groupby("Patient ID")
        .apply(lambda g: list(zip(g["Slide"], g["Bbox coordinates normalized (X, Y, W, H)"])))
        .to_dict()
    )
        all_data = []
        print("mode", mode)
        for path in data_paths:
            with open(path, "r") as f:
                data = json.load(f)
                for i, dp in enumerate(data):             
                    if "A4" not in dp:
                        print(f"Skipping data point {i} in {path} due to missing A4")
                        continue
                    all_data.append((path, i, dp["A4"].lower().replace("-", " ")))
        random.shuffle(all_data)
        counter=defaultdict(int)
        if n_sample > 0:
            all_data = all_data[:n_sample]
        for _,_,a4 in all_data:
            counter[a4]+=1
        print("A4 distribution before balancing:", dict(counter))
        self.sample_indices = [(p, i,a) for p, i, a in all_data]
        counter=defaultdict(int)
        random.shuffle(self.sample_indices)
        for _,_,a4 in self.sample_indices:
            counter[a4]+=1
        self.base_transform= get_base_transform(spatial_size=[-1, self.height, self.width])

    def _load_image(self, patient_id, slice_order):
        image_paths = [os.path.join(self.img_dir, patient_id, f"{s}.pkl") for s in slice_order]
        for p in image_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image slice not found: {p}")
        image =convert_list_slice_paths_to_3d(image_paths)
        return image

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        path, sample_idx,_ = self.sample_indices[idx]
        with open(path, "r") as f:
            data_point = json.load(f)[sample_idx]
        image = self._load_image(data_point["Patient ID"], data_point["slice order"])
        image = self.base_transform({"image": image})["image"]
        image=None
        answer = ""
        match_slice_idx=None
        match_bbox=[]
        slice_order = data_point["slice order"]
        significant_slices_per_patient=self.significant_slices_data.get(data_point["Patient ID"], [])
        if significant_slices_per_patient:
            match_slice_tuple=[(s, b, slice_order.index(s)) 
                                for s, b in significant_slices_per_patient 
                                        if s in slice_order]
            if len(match_slice_tuple)>0:
                match_slice_idx=match_slice_tuple[0][2]
                raw_bbox = match_slice_tuple[0][1]
                if raw_bbox:
                    match_bbox = normalize_bbox(raw_bbox)
        # ground_truth=json.dumps({
        #     "slice_index": match_slice_idx,
        #     "bounding_boxes": match_bbox,
        #     "description":data_point['A2'],
        #     "score":data_point['A3'],
        #     "severity":data_point['A4'],
        #     } ,ensure_ascii=False)
        A_1_part="No significant lesion"
        if match_slice_idx is not None:
            A_1_part=f"Significant lesion at slice index {match_slice_idx}."
        A_3_part= ", ".join(f"{k}={v}" for k, v in data_point['A3'].items())
        ground_truth = (
                f"A1: {A_1_part};\n"
                f"A2: {data_point['A2']};\n"
                f"A3: {A_3_part};\n"
                f"A4: {data_point['A4']}"
            )
        if self.mode !="test":
            answer = ground_truth
     
        question=data_point["Q4"][0]

        return {
            "image": image,
            "ground_truth": ground_truth,
            "slice_order": data_point["slice order"],
            "P_ID": data_point["Patient ID"],
            "question":question,
            "Q1": data_point["Q1"],
            "Q2": data_point["Q2"],
            "Q3": data_point["Q3"],
            "Q4": data_point["Q4"],
            "A1": data_point["A1"],
            "A2": data_point["A2"],
            "A3": data_point["A3"],
            "A4": data_point["A4"],
            "answer": answer,
        }

def normalize_bbox(raw_bbox: str):
    """
    Convert raw_bbox string like:
    "[0.06, 0.5, 0.29, 0.67],[0.69, 0.5, 0.92, 0.65],..."
    into canonical list of list of floats:
    [[0.06, 0.5, 0.29, 0.67], [0.69, 0.5, 0.92, 0.65], ...]
    """
    if not raw_bbox or not isinstance(raw_bbox, str):
        return []

    # Wrap in brackets to make a single list
    raw_bbox_wrapped = f"[{raw_bbox}]"

    try:
        bbox_list = ast.literal_eval(raw_bbox_wrapped)
    except Exception:
        return []

    # Ensure floats
    result = []
    for b in bbox_list:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            result.append([float(v) for v in b])
    return result
      
if __name__ == "__main__":
    import yaml
    from tqdm import tqdm
    from transformers import AutoTokenizer
    train_val_dir= "pseudo_3d/32_all_slices/0fa350fe-9eb2-4b61-916f-5a29e322f6ab/train/data"
    test_dir= "pseudo_3d/32_all_slices/0fa350fe-9eb2-4b61-916f-5a29e322f6ab/test/data"
    train_image="clean_data_s_chain/train/image"
    test_image="clean_data_s_chain/test/image"
    test_path=[os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')]
    train_path=[os.path.join(train_val_dir, f) for f in os.listdir(train_val_dir) if f.endswith('.json')]
    significant_slice_path="significant_slice.csv"
    train_set = TracDatasetCoT( train_path, train_image,mode="train", n_sample=-1)
    test_set = TracDatasetCoT( test_path, test_image,mode="train", n_sample=-1)
    train_save_path="train_set.jsonl"
    test_save_path="test_set.jsonl"
    train_data=[]
    test_data=[]
    for i in range(len(train_set)):
        sample = train_set[i]
        slice_order = sample["slice_order"]
        P_ID = sample["P_ID"]
        question = sample["question"]
        answer = sample["answer"]

        print("question:", question)
        print("answer:", answer)
    # with open(train_save_path, "w", encoding="utf-8") as f:
    #     for sample in tqdm(train_set, desc="Saving train set"):
    #         slice_order = sample["slice_order"]
    #         P_ID = sample["P_ID"]
    #         question = sample["question"]
    #         answer = sample["answer"]

    #         record = {
    #             "slice_order": slice_order,
    #             "P_ID": P_ID,
    #             "question": question,
    #             "answer": answer
    #         }
    #         train_data.append(record)
    #         f.write(json.dumps(record, ensure_ascii=False) + "\n")  # JSONL line

    # # Save test set
    # with open(test_save_path, "w", encoding="utf-8") as f:
    #     for sample in tqdm(test_set, desc="Saving test set"):
    #         slice_order = sample["slice_order"]
    #         P_ID = sample["P_ID"]
    #         question = sample["question"]
    #         answer = sample["answer"]

    #         record = {
    #             "slice_order": slice_order,
    #             "P_ID": P_ID,
    #             "question": question,
    #             "answer": answer
    #         }
    #         test_data.append(record)
    #         f.write(json.dumps(record, ensure_ascii=False) + "\n")
    