import json 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
import cv2
import pickle
import random
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_grid(single_slice_data_folder,data_paths, base_image_path, save_dir,  scale_factor=1.5, dpi=400,
                   cell_h=496, cell_w=256, label_px=24):
  
    save_image_folder = os.path.join(save_dir, "images")    
    if os.path.exists(save_image_folder):
        shutil.rmtree(save_image_folder)
    
    save_significant_slice_dir=os.path.join(save_image_folder, "significant_slices")
    save_grid_dir=os.path.join(save_image_folder, "grid")
    save_image_3d_dir=os.path.join(save_image_folder, "image_3d")
    if os.path.exists(save_significant_slice_dir):
        shutil.rmtree(save_significant_slice_dir)
    if os.path.exists(save_grid_dir):
        shutil.rmtree(save_grid_dir)
    if os.path.exists(save_image_3d_dir):
        shutil.rmtree(save_image_3d_dir)
    
    os.makedirs(save_significant_slice_dir, exist_ok=True)
    os.makedirs(save_image_folder, exist_ok=True)
    os.makedirs(save_grid_dir, exist_ok=True)
    os.makedirs(save_image_3d_dir, exist_ok=True)

    slice_detail_paths=[os.path.join(single_slice_data_folder,f)  for f in os.listdir(single_slice_data_folder) if f.endswith('.json')]
    slice_details=defaultdict(dict)
    for path in slice_detail_paths:
        with open(path, "r") as f:
            data = json.load(f)
        p_id=path.split("/")[-1].replace(".json","")
        for dp in data:
            if p_id not in slice_details:
                slice_details[p_id]={}
            slice_name=dp["Slide"]  
            slice_details[p_id][slice_name]=dp
    # print("slice detail",slice_de)
    for k,v in slice_details.items():
        print(f"Patient {k} has {len(v)} slice details")
        keys=list(v.keys())
        keys.sort()
        # for sk,sv in v.items():
        #     print(f"  Slice {sk}: A2={sv.get('A2','')}, A3={sv.get('A3','')}, A4={sv.get('A4','')}")
    
    significant_slice_path = "significant_slice.csv"
    df = pd.read_csv(significant_slice_path)
    significant_slices_data = (
        df.dropna(subset=["Slide"])
          .groupby("Patient ID")
          .apply(lambda g: list(zip(g["Slide"], g["Bbox coordinates normalized (X, Y, W, H)"])))
          .to_dict()
    )

    all_data = []
    for path in data_paths:
        with open(path, "r") as f:
            data = json.load(f)
        for i, dp in enumerate(data):
            if "A4" not in dp:
                continue
            all_data.append(dp)
    random.seed(42)
    random.shuffle(all_data)

    output_records = []
    for i, data_point in enumerate(all_data):
        p_id = data_point["Patient ID"]
        slice_order = data_point["slice order"]
        image_save_path_3d=os.path.join(save_image_3d_dir, f"image_{i}_3d.npy")
        image_save_path_grid=os.path.join(save_grid_dir, f"grid_{i}.png")
        A1=json.dumps( data_point.get("A1", ""))
        A3=json.dumps( data_point.get("A3", ""))
        sample = {
            "Patient ID": p_id,
            "Q1": data_point.get("Q1", [""])[0] if data_point.get("Q1") else "",
            "A1": A1,
            "Q2": data_point.get("Q2", [""])[0] if data_point.get("Q2") else "",
            "A2": data_point.get("A2", ""),
            "Q3": data_point.get("Q3", [""]*5)[0],
            "A3": A3,
            "Q4": data_point.get("Q4", [""]*5)[0],
            "A4": data_point.get("A4", ""),
            "sample_significant_slice_file_name": None,
            "image_grid_file_name": image_save_path_grid,
            "3d_image":f"grid_{i}.png",
            "chain_of_thought_slice_index":""
            "chain_of_thought_slice_name"
        }

        # Prepare significant slice indexes
        significant_slices_per_patient = significant_slices_data.get(p_id, [])
        match_slice_indexes = []
        cot_sample={}
        cot_sample_with_slice_info={}
        if significant_slices_per_patient:
            
            slice_idx_map = {s: idx for idx, s in enumerate(slice_order)}
            match_slice_tuple = [
                (s, b, slice_idx_map[s])
                for s, b in significant_slices_per_patient
                if s in slice_idx_map
            ]
            if match_slice_tuple:
                # print("pid", p_id, "match slices:", match_slice_tuple)
                match_slice_indexes = [t[2] for t in match_slice_tuple]
                cot_sample["represent_slices"]= match_slice_indexes
                cot_sample["bounding_boxes"] = {t[2]: t[1] for t in match_slice_tuple}
                cot_sample["slice_wise_reason"] = {t[2]: slice_details[p_id][t[0]].get("A2", "") for t in match_slice_tuple}
                cot_sample["slice_wise_rating"]={t[2]: get_visualize_rating( slice_details[p_id][t[0]].get("A3", "")) for t in match_slice_tuple}
                cot_sample["scan_wise_reason"]= get_scan_wise_visual_rating( [slice_details[p_id][t[0]].get("A3", "") for t in match_slice_tuple])
                cot_sample["scan_wise_diagnosis"]= get_final_diagnosis_mapping( [slice_details[p_id][t[0]].get("A4", "") for t in match_slice_tuple])
                
                
                cot_sample["represent_slices_text"]= [t[0] for t in match_slice_tuple]
                cot_sample["bounding_boxes_text"] = {t[0]: t[1] for t in match_slice_tuple}
                cot_sample["slice_wise_reason_text"] = {t[0]: slice_details[p_id][t[0]].get("A2", "") for t in match_slice_tuple}
                cot_sample["slice_wise_rating_text"]={t[0]: get_visualize_rating( slice_details[p_id][t[0]].get("A3", "")) for t in match_slice_tuple}
                
                sample["chain_of_thought_slice_index"]=prepare_cot_column(cot_sample)
                sample["chain_of_thought_slice_name"]=prepare_cot_column_with_slice_info(cot_sample)
                first_sig_name = match_slice_tuple[0][0]
                significant_slice_path = os.path.join(base_image_path, p_id, f"{first_sig_name}.pkl")
                with open(significant_slice_path, "rb") as f:
                    sig_image = pickle.load(f)
                os.makedirs(os.path.join(save_significant_slice_dir, p_id), exist_ok=True)
                sig_image = sig_image.squeeze()
                sig_save_path = os.path.join(save_significant_slice_dir, p_id, f"{first_sig_name}.png")
                sample["sample_significant_slice_file_name"] = sig_save_path
                plt.imsave(sig_save_path, sig_image, cmap="gray", dpi=200)
                # remember to remove this
                
        # Visualize and save
        slice_paths=[os.path.join(base_image_path, p_id, f"{s}.pkl") for s in slice_order]
        
        
        save_image_grid(  
            sample=data_point,
            special_idx=match_slice_indexes,
            image_paths=slice_paths,
            save_path=image_save_path_grid,
            scale_factor=scale_factor,
            dpi=dpi,
            cell_h=cell_h,
            cell_w=cell_w,
            label_px=label_px
        )
        save_image_3d(image_paths=slice_paths, save_path=image_save_path_3d)
        output_records.append(sample)
        if i==15:
            break

    output_df = pd.DataFrame(output_records)
    save_file=os.path.join(save_dir, "data.parquet")
    output_df.to_parquet(save_file, index=False)
    print("Saved metadata to:", save_file)


def prepare_cot_column(sample):
    text="Q1: Representative slice selection \n"
    text+=f"A1: {sample.get('represent_slices',[])}\n"
    text+="Q2: Bounding box for representative slice \n"
    text+=f"A2: {sample.get('bounding_boxes',[])}\n"
    text+="Q3: Slice-wise reasoning\n"
    text+=f"A3: {sample.get('slice_wise_reason',[])}\n"
    text+="Q4: Slice-wise visual rating\n"
    text+=f"A4: {sample.get('slice_wise_rating',[])}\n"
    text+=f"Q5: Scan-wise composite reasoning\n"
    text+=f"A5: {sample.get('scan_wise_reason',[])}\n"
    text+=f"Q6: Scan-wise final diagnosis\n"
    text+=f"A6: {sample.get('scan_wise_diagnosis',[])}\n"
    return text

def prepare_cot_column_with_slice_info(sample):
    text="Q1: Representative slice selection \n"
    text+=f"A1: {sample.get('represent_slices_text',[])}\n"
    text+="Q2: Bounding box for representative slice \n"
    text+=f"A2: {sample.get('bounding_boxes_text',[])}\n"
    text+="Q3: Slice-wise reasoning\n"
    text+=f"A3: {sample.get('slice_wise_reason_text',[])}\n"
    text+="Q4: Slice-wise visual rating\n"
    text+=f"A4: {sample.get('slice_wise_rating_text',[])}\n"
    text+=f"Q5: Scan-wise composite reasoning\n"
    text+=f"A5: {sample.get('scan_wise_reason',[])}\n"
    text+=f"Q6: Scan-wise final diagnosis\n"
    text+=f"A6: {sample.get('scan_wise_diagnosis',[])}\n"
    return text
    
def rgb_to_grayscale(img_rgb):
    """Convert (H, W, 3) RGB to (H, W) grayscale using standard weights."""
    return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]) 

def save_image_3d(image_paths, save_path):
    img_3d = []
    for image_path in image_paths:
        with open(image_path, "rb") as f:
            image = pd.read_pickle(f)
        if image.ndim == 3:
            image = rgb_to_grayscale(image)

        image = image.squeeze()
        img_3d.append(image)
    img_3d = np.stack(img_3d, axis=0).astype(np.float32)
    np.save(save_path, img_3d)
    print(f"Saved 3D data to {save_path}")

def get_visualize_rating(s: str) -> str:
    """Merge A3 scoring data."""
    result = {"GCA": 0, "Koedam": 0, "MTA": 0}
    if s:
        try:
            for pair in [p.strip() for p in s.split(",")]:
                key, value = pair.split("=")
                result[key.strip()] = max(result.get(key.strip(), 0), int(value.strip()))
            return ", ".join([f"{k}={v}" for k, v in result.items()])
        except Exception as e:
            print.error(f"Error merge_visualize_rating_data {s}: {e}")
    return ""

def get_final_diagnosis_mapping(ls:list):  
        degree_levels={
                "non": "Non-Dementia",
                "mild": "Mild-Dementia",
                "moderate": "Moderate-Dementia"
            }

        if not ls:
            logger.warning("Empty A4 list, defaulting to Non-Dementia")
            return "Non-Dementia"

        cleaned_list = [str(s) for s in ls if s is not None]
        for level,description in reversed(degree_levels.items()):
        
            for s in cleaned_list:
                if level.lower() in s.lower():
                    logger.info(f"match s {s} for {level}")
                    return description

        logger.warning(f"No dementia level found in A4, defaulting to Non-Dementia: {cleaned_list}")
        return "Non-Dementia"

def get_scan_wise_visual_rating(ls:list):
    result={"GCA":0,"Koedam":0,"MTA":0}
    if len(ls)>0:
        for s in ls:
            try:
                for pair in [p.strip() for p in s.split(",")]:
                    key, value = pair.split("=")
                    result[key.strip()] = max(result.get(key.strip(), 0), int(value.strip()))
            except Exception as e:
                print.error(f"Error merge_scan_wise_visual_rating {s}: {e}")
        return ", ".join([f"{k}={v}" for k, v in result.items()])
    print("Empty list for scan wise visual rating")
    return ""

def scale_to_cell(image, cell_h, cell_w, keep_aspect=True):
    if image.ndim == 2:
        h, w = image.shape
        c = None
    else:
        h, w, c = image.shape

    if keep_aspect:
        scale = min(cell_w / w, cell_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if c is None:
            canvas = np.zeros((cell_h, cell_w), dtype=resized.dtype)
            y0 = (cell_h - new_h) // 2
            x0 = (cell_w - new_w) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        else:
            canvas = np.zeros((cell_h, cell_w, c), dtype=resized.dtype)
            y0 = (cell_h - new_h) // 2
            x0 = (cell_w - new_w) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w, :] = resized

        return canvas

    else:
        # force resize (aspect ratio ignored)
        return cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_CUBIC)
def save_image_grid(
    sample: dict,
    special_idx: list,
    image_paths: list,
    save_path: str,
    scale_factor=1.5,
    dpi=400,
    cell_h=496,
    cell_w=256,
    label_px=24,
):
    import math

    p_id = sample["Patient ID"]
    slice_order = sample["slice order"]
    dementia_level = sample["A4"]

    max_cols = 8
    n_images = len(image_paths)

    n_cols = min(max_cols, max(1, math.ceil(math.sqrt(n_images))))
    n_rows = math.ceil(n_images / n_cols)

    print("Saving:", save_path)

    figsize = (
        n_cols * cell_w * scale_factor / dpi,
        n_rows * (cell_h * scale_factor + label_px) / dpi,
    )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

    axes = np.atleast_1d(axes).ravel()

    fig.suptitle(
        f"Patient ID: {p_id} | Dementia Level: {dementia_level}",
        fontsize=12,
    )

    try:
        for i, image_path in enumerate(image_paths):
            ax = axes[i]

            with open(image_path, "rb") as f:
                image = pd.read_pickle(f)

            image = image.squeeze()

            new_h = int(image.shape[0] * scale_factor)
            new_w = int(image.shape[1] * scale_factor)

            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            image_cell = scale_to_cell(
                image,
                cell_h=int(cell_h * scale_factor),
                cell_w=int(cell_w * scale_factor),
                keep_aspect=True,
            )

            ax.imshow(image_cell, cmap="gray", aspect="equal")
            ax.axis("off")

            # ✅ correct label position
            ax.text(
                image_cell.shape[1] / 2,
                -label_px * 0.2,
                str(slice_order[i]),
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=False,
            )

            if i in special_idx:
                ax.add_patch(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=ax.transAxes,
                        fill=False,
                        linewidth=2,
                        edgecolor="red",
                    )
                )

        for j in range(len(image_paths), len(axes)):
            axes[j].axis("off")

        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0, hspace=0.15)
        plt.savefig(save_path, dpi=dpi)
        plt.close(fig)

        print("Saved figure to:", save_path)

    except Exception as e:
        plt.close(fig)
        print(f"Error processing patient {p_id}: {e}")
    
if __name__ == "__main__":
    data_path="pseudo_3d/32_all_slices/edf3c478-aaff-43ec-8d51-7ef698ad1244"
    train_path= data_path + "/train/data/"
    test_path= data_path + "/test/data/"
    train_image_path="clean_data_s_chain/train/image"
    test_image_path="clean_data_s_chain/test/image"
    train_slice_dir="clean_data_s_chain/train/data"
    # train_image_path="clean_data_s_chain/train/image_with_bboxes"
    # test_image_path="clean_data_s_chain/test/image_with_bboxes"
    significant_slice_path="significant_slice.csv"
    # visualize_and_save(train_path,train_image_path,significant_slice_path,mode="train")
    # all_train_files = [os.path.join(train_path,f)  for f in os.listdir(train_path) if f.endswith('.json')]
    # all_test_files = [os.path.join(test_path,f)  for f in os.listdir(test_path) if f.endswith('.json')]
    all_train_files = ["pseudo_3d/32_all_slices/edf3c478-aaff-43ec-8d51-7ef698ad1244/train/data/OAS1_0002.json"]
    visualize_grid(train_slice_dir,
                    all_train_files,
                   train_image_path,
                   save_dir="../viz_data/train",
    )
    # visualize_grid(all_test_files,
    #                test_image_path,
    #                save_dir="../viz_data/test",
    #                save_image_folder="../viz_data/test/images/grid",
    #                  save_significant_slice_dir="../viz_data/test/images/significant_slices"
    #                )
    # sample={
    #     "Patient ID": "OAS1_0002",
    #     "slice order": [f"mpr-1_{i}" for i in range(32)],
    #     "A4": "moderate dementia"
    # }
    # save_image_grid(
    #     sample=sample,
    #     special_idx=[5],
    #     base_image_path=train_image_path,
    #     save_path="test_grid.png",
    #     scale_factor=1.5,
    #     dpi=400,
    #     n_cols=8,
    #     cell_h=496,
    #     cell_w=256,
    #     label_px=24
    # )
    # pass