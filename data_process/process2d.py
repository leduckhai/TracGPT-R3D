import os
import shutil
import pickle
import json
import numpy as np
from datasets import load_dataset
from collections import defaultdict
<<<<<<< Updated upstream
import json 

ds = load_dataset("tungvu3196/vlm-project-with-images-with-bbox-images-v6")
output_folder="./clean_data"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
=======
from tqdm import tqdm
import logging
>>>>>>> Stashed changes

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_cache():
    """Clear all possible cache locations"""
    cache_locations = [
        "/tmp/hf_cache",
        "/tmp/hf_cache_new", 
        os.path.expanduser("~/.cache/huggingface/datasets"),
        "/root/.cache/huggingface/datasets"
    ]
    
    for cache_dir in cache_locations:
        if os.path.exists(cache_dir):
            try:
                logger.info(f"Removing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir)
            except Exception as e:
                logger.warning(f"Failed to remove {cache_dir}: {str(e)}")

def process_dataset(dataset, output_folder, split_name):
    """Process a single dataset split"""
    clients_map = {}
    patients = defaultdict(list)
    
    annot_root = os.path.join(output_folder, split_name, "image_with_bboxes")
    image_root = os.path.join(output_folder, split_name, "image")
    data_folder = os.path.join(output_folder, split_name, "data")
    
    os.makedirs(annot_root, exist_ok=True)
    os.makedirs(image_root, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    
    for i in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
        try:
            sample = dataset[i]
            patient_id = sample["Patient ID"]
            slide = sample["Slide"]
            
            # Create patient directories
            annot_folder = os.path.join(annot_root, patient_id)
            image_folder = os.path.join(image_root, patient_id)
            os.makedirs(annot_folder, exist_ok=True)
            os.makedirs(image_folder, exist_ok=True)
            
            # Save images
            image_path = os.path.join(image_folder, f"{slide}.pkl")
            annot_path = os.path.join(annot_folder, f"{slide}.pkl")
            
            with open(image_path, "wb") as f:
                pickle.dump(np.array(sample["image"]), f)
            
            with open(annot_path, "wb") as f:
                pickle.dump(np.array(sample["image_with_bboxes"]), f)
            
            # Remove large fields before saving metadata
            processed_sample = {k: v for k, v in sample.items() 
                               if k not in ["image", "image_with_bboxes"]}
            patients[patient_id].append(processed_sample)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            continue
    
    # Save patient metadata
    for patient_id, data in patients.items():
        patient_data_path = os.path.join(data_folder, f"{patient_id}.json")
        with open(patient_data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # Clear cache
    clear_cache()
    
    # Set up clean cache directory
    cache_dir = "/tmp/hf_cache_clean"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load dataset
    try:
        logger.info("Loading dataset...")
        ds = load_dataset(
            "tungvu3196/vlm-project-with-images-with-bbox-images-v6", 
            cache_dir=cache_dir,
            download_mode="force_redownload"
        )
        logger.info("Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return
    
    # Prepare output folder
    output_folder = "./clean_data"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Process each split
    for split_name in ds.keys():
        logger.info(f"Processing split: {split_name}")
        process_dataset(ds[split_name], output_folder, split_name)

if __name__ == "__main__":
    main()