
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
base_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/image_with_bboxes/OAS1_0002"

import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

def viz_all(save_folder, base_dir):
    """
    Load RGB images from pickle files and save as PNG images
    
    Args:
        save_folder: Directory to save output images
        base_dir: Directory containing pickle files with RGB images (H,W,3)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    
    # Get all pickle files
    file_pkls = [f for f in os.listdir(base_dir) if f.endswith('.pkl')]
    
    for i, file_pkl in enumerate(tqdm(file_pkls, desc="Processing images")):
        try:
            # Load data
            with open(os.path.join(base_dir, file_pkl), "rb") as f:
                img_data = pickle.load(f)
            
            # Convert to PIL Image (assuming img_data is numpy array with shape [H,W,3])
            if isinstance(img_data, np.ndarray):
                if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                    # Convert from float (0-1) to uint8 (0-255)
                    img_data = (img_data * 255).astype(np.uint8)
                elif img_data.dtype == np.uint8:
                    # Already in correct format
                    pass
                else:
                    raise ValueError(f"Unsupported image dtype: {img_data.dtype}")
                
                img = Image.fromarray(img_data)
            else:
                raise ValueError("Loaded data is not a numpy array")
            
            # Save image
            save_path = os.path.join(save_folder, f"{os.path.splitext(file_pkl)[0]}.png")
            img.save(save_path)
            
        except Exception as e:
            print(f"Error processing {file_pkl}: {str(e)}")
            continue

# Example usage
viz_all("output_images", base_dir)