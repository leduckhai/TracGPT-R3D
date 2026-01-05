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
    def __init__(self, data_paths, image_path,sigfinicant_slice_path, mode="train", n_sample=-1):
        random.seed(68)  
        self.img_dir = image_path
        self.sample_indices = []    
        self.mode = mode
        self.height = 256
        self.width = 256
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
        image =convert_list_slice_paths_to_3d(image_paths)
        return image

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        print("idx",idx)
        path, sample_idx,_ = self.sample_indices[idx]
        with open(path, "r") as f:
            data_point = json.load(f)[sample_idx]
        # image = self._load_image(data_point["Patient ID"], data_point["slice order"])
        answer = ""
        if self.mode == "train":
            slice_order = data_point["slice order"]
            significant_slices_per_patient=self.significant_slices_data.get(data_point["Patient ID"], [])
            # image = self._load_image(data_point["Patient ID"], slice_order) 
            step_1="Identify lesion region"
            step_2="What observable particularities characterize this lesion?"
            step_3="What is the severity index of this lesion?"
            step_4="How significant is the disease burden?"
            # step_1_answer="No significant slice index and bounding box found"
            match_slice_idx=""
            match_bbox=""
            if significant_slices_per_patient:
                match_slice_tuple=[(s, b, slice_order.index(s)) 
                                    for s, b in significant_slices_per_patient 
                                            if s in slice_order]
                if len(match_slice_tuple)>0:
                    match_slice_idx=match_slice_tuple[0][2]
                    match_slice=match_slice_tuple[0][0]
                    match_bbox=match_slice_tuple[0][1]
                    print("Found significant slice:", match_slice, "at index:", match_slice_idx)
                    if match_slice:
                        step_1_answer=f"Slice index {match_slice_idx}, bounding box [{match_bbox}]"
       
            question=f"""
            step 1: {step_1}
            step 2: {step_2}
            step 3: {step_3}
            step 4: {step_4}
            """
            answer = {
                "slice_index": match_slice_idx,
                "bounding_box": match_bbox,
                "description":data_point['A2'],
                "score":data_point['A3'],
                "severity":data_point['A4']
            } 
            answer=json.dumps(answer,ensure_ascii=False)
        
        # image = self._load_image(data_point["Patient ID"], slice_order)
        # image = self.base_transform({"image": image})["image"]
        # if self.train_transform:
        #     image = self.train_transform({"image": image})["image"]
        #     image=image.squeeze(0)
            
        idx=random.randrange(len(data_point["Q4"]))
        # question="How grievous is this medical situation?"
        return {
            # "image": image,
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
        
if __name__ == "__main__":
    import yaml
    from transformers import AutoTokenizer
    train_val_dir= "pseudo_3d/32_all_slices/0fa350fe-9eb2-4b61-916f-5a29e322f6ab/train/data"
    test_dir= "pseudo_3d/32_all_slices/0fa350fe-9eb2-4b61-916f-5a29e322f6ab/test/data"
    train_image="clean_data_s_chain/train/image"
    test_path=[os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.json')]
    train_path=[os.path.join(train_val_dir, f) for f in os.listdir(train_val_dir) if f.endswith('.json')]
    significant_slice_path="significant_slice.csv"
    dataset = TracDatasetCoT( train_path, train_image,significant_slice_path, mode="train", n_sample=-1)
    # for i in range(len(dataset)):
    #     # print(dataset[i])
    #     sample=dataset[i]
    print("len dataset", len(dataset))
    for i in range(50):
        sample = dataset[i]
        print("sample",sample["answer"])
        # print(sample["image"].shape, sample["P_ID"], sample["A4"], sample["Q1"], sample["A1"])
  