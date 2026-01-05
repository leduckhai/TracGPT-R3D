import os 
import json
from collections import defaultdict
import shutil
import re
import pandas as pd

test_data_raw_path="clean_data_s_chain/test/data"
process_data_path="pseudo_3d/32_all_slices/7cc980de-e632-4f11-9805-65acc291fcb7/test/data"

# test_raw_output_file="test_raw.json"
test_raw_output_dir="test_raw_outputs"
test_processed_output_dir="test_processed_outputs"
# test_processed_output_file="test_processed.json"
def group_files(file_list):
    group=defaultdict(list)
    sorted = sort_files(file_list)
    for f in sorted:
        id= f.split("_")[0].split("-")[1]
        group[id].append(f)
    return list(group.values())

def sort_files(file_list):
    def sort_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))[::-1]
        return numbers 

    sorted_files = sorted(file_list, key=sort_key)
    return sorted_files
def process_raw_data():
    all_raw_files = os.listdir(test_data_raw_path)
    print("All processed files:", all_raw_files)
    for file in all_raw_files:
        print("Processing file:", file)
        with open(os.path.join(test_data_raw_path, file), "r") as f:
            raw_data = json.load(f)
        saved_path=os.path.join(test_raw_output_dir,file)
        print("Saved processed file path:", saved_path)   
        unique_status=defaultdict(list)
        print("Unique A4 statuses in file:", unique_status)
        unique_bbox_status=defaultdict(list)
        for item in raw_data:
            s=item["A3"]
            print("Processing A3 status:", s)
            try:
                for pair in [p.strip() for p in s.split(",")]:
                    key, value = pair.split("=")
                    dict_key=key.strip()+"_"+value.strip()
                    unique_bbox_status[dict_key].append(item["Slide"])
            except Exception as e:
                print(f"Error merging A3 data {s}: {e}")
            unique_status[item["A4"]].append(item["Slide"])
        unique_bbox_status={k:group_files(v) for k,v in unique_bbox_status.items() if v}
        save_data={
            "unique_A3_status": dict(unique_bbox_status),
            "unique_A4_status": dict(unique_status),
        }
        with open(saved_path, "w") as f:
            json.dump(save_data, f, indent=4)
def process_processed_data():
    all_processed_files = os.listdir(process_data_path)
    print("All processed files:", all_processed_files)
    for file in all_processed_files:
        print("Processing file:", file)
        with open(os.path.join(process_data_path, file), "r") as f:
            processed_data = json.load(f)
        saved_path=os.path.join(test_processed_output_dir,file)
        print("Saved processed file path:", saved_path)   
        unique_status=defaultdict(list)
        print("Unique A4 statuses in file:", unique_status)
        unique_bbox_status=defaultdict(list)
        slice_order_idx={}
        for i,item in enumerate(processed_data):
            slice_order_idx[i]=item["slice order"]
            s=item["A3"]
            print("Processing A3 status:", s)
            try:
                for k,v in s.items():
                    dict_key=str(k)+"_"+str(v)
                    unique_bbox_status[dict_key].append(i)
            except Exception as e:
                print(f"Error merging A3 data {s}: {e}")
            unique_status[item["A4"]].append(i)
        
        save_data={
            "unique_A3_status": dict(unique_bbox_status),
            "unique_A4_status": dict(unique_status),
            "slice_order_idx": slice_order_idx,
        }
        with open(saved_path, "w") as f:
            json.dump(save_data, f, indent=4)
def map_significant_slices():
    significant_slice_path="significant_slice.csv"
    if not os.path.exists(significant_slice_path):
        print(f"Significant slice file not found: {significant_slice_path}")
        return

    df = pd.read_csv(significant_slice_path)
    result = (
    df.dropna(subset=["Slide"])        # optional: remove rows where Slide is NaN
      .groupby("Patient ID")
     .apply(lambda g: list(zip(g["Slide"], g["Bbox coordinates normalized (X, Y, W, H)"])))
      .apply(list)
      .to_dict()
)
    
    sample_path="pseudo_3d/32_all_slices/0fa350fe-9eb2-4b61-916f-5a29e322f6ab/train/data/OAS1_0002.json"
    with open(sample_path, "r") as f:
        sample_data = json.load(f)
    p_id="OAS1_0002"
    significant_slices = result.get(p_id, [])
    print(f"Significant slices for patient {p_id}:", significant_slices)
    for item in sample_data:
        slice_order=item["slice order"]
        match_slice_tuple=[s for s, _ in significant_slices if s in slice_order]
        print("Matching significant slices in slice order:", match_slice_tuple)      
    process_data_path=""
if __name__=="__main__":
    # if os.path.exists(test_raw_output_dir):
    #     shutil.rmtree(test_raw_output_dir)
    # if os.path.exists(test_processed_output_dir):
    #     shutil.rmtree(test_processed_output_dir)
    # os.makedirs(test_raw_output_dir, exist_ok=True)
    # os.makedirs(test_processed_output_dir, exist_ok=True)
    # process_raw_data()
    # process_processed_data()
    map_significant_slices()

