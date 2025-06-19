import pickle

# path="/home/ducnguyen/sync_local/repo/TracGPT/images/train/image_with_bboxes_OAS1_0373_mpr-4_160.pkl"
# with open(path, "rb") as f:
#     data=pickle.load(f)
#     print("data", data.shape)
viz_files=[
    "/home/ducnguyen/sync_local/repo/TracGPT/clean_data_3d/image_3612ae26-9711-42f1-a4aa-9d91aba23e70/train/image/OAS1_0031.pkl"
]

import pickle
import nibabel as nib
import numpy as np
import shutil
import os 

# for path in viz_files:
#     base_name=path.split("/")[-1].split(".")[0]
#     with open(path, 'rb') as f:
#         volume = pickle.load(f)  


#     nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
#     nib.save(nifti_img, f'{base_name}.nii.gz')

# file_pattern=f"/home/ducnguyen/sync_local/repo/TracGPT/images/train/image_OAS1_0137_mpr-3_<number>.pkl"


# splits=["train","test"]
# source_data_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data"
# source_img_dir="/home/ducnguyen/sync_local/repo/TracGPT/images"

# for split in splits:
#     target_img_dir=os.path.join(base_image_dir,split)
#     output_folder=os.path.join(base_data_dir,split)
#     subdir_img=os.path.join(target_img_dir,"image")
#     subdir_annot=os.path.join(target_img_dir,"image_with_bboxes")

#     if os.path.exists(output_folder):
#         shutil.rmtree(output_folder)
#     if os.path.exists(target_img_dir):
#         shutil.rmtree(target_img_dir)

#     os.makedirs(output_folder, exist_ok=True)
#     os.makedirs(target_img_dir, exist_ok=True)
#     os.makedirs(subdir_img, exist_ok=True)
#     os.makedirs(subdir_annot, exist_ok=True)

#     patient_dirs=os.listdir(os.path.join(source_data_dir,split))
#     save_data_path=os.path.join(output_folder,"data.csv")
#     clients=[]

#     for i in range(len(patient_dirs)):
#         p_dir=patient_dirs[i]
#         slide_dir=os.path.join(source_data_dir,split,p_dir)
#         slide_paths= [os.path.join(slide_dir, f) for f in os.listdir(slide_dir)]



import glob
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
from PIL import Image
import math 



def organize_slices(input_folder, output_folder):
    # Match filenames like image_with_bboxes_OAS1_0002_mpr-2_130.pkl
    pattern = re.compile(r'^(image(?:_with_bboxes)?)_(OAS1_\d{4})_(mpr-\d+_\d+)\.(pkl|jpg|png|jpeg)$')

    for filename in os.listdir(input_folder):
        match = pattern.match(filename)
        if match:
            file_type = match.group(1)      # e.g., image, image_with_bboxes
            patient_id = match.group(2)     # e.g., OAS1_0002
            slice_name = match.group(3)     # e.g., mpr-2_130
            ext = match.group(4)            # file extension

            # Create output directory
            dest_folder = os.path.join(output_folder, file_type, patient_id)
            os.makedirs(dest_folder, exist_ok=True)

            # Build destination file path
            dst_filename = f"{slice_name}.{ext}"
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(dest_folder, dst_filename)

            shutil.copy2(src_path, dst_path)  # use shutil.move(...) if needed

            print(f"Saved: {dst_filename} → {dst_path}")
        else:
            print(f"Skipped (pattern not matched): {filename}")

input_dir = "/path/to/slices"
output_dir = "/path/to/organized"

train_src_dir="/home/ducnguyen/sync_local/repo/TracGPT/images/train"
train_dst_dir="/home/ducnguyen/sync_local/repo/TracGPT/image_group/train"
test_src_dir="/home/ducnguyen/sync_local/repo/TracGPT/images/test"
test_dst_dir="/home/ducnguyen/sync_local/repo/TracGPT/image_group/test"

# organize_slices(train_src_dir, train_dst_dir)
# organize_slices(test_src_dir, test_dst_dir)

patient_dirs="/home/ducnguyen/sync_local/repo/TracGPT/image_group/train/image/OAS1_0031"
patient_id=patient_dirs.split("/")[-1]
# folder = "/home/ducnguyen/sync_local/repo/TracGPT/images/train/"
# pattern = re.compile(r"image_OAS1_0137_mpr-3_\d+\.pkl")
all_files = glob.glob(os.path.join(patient_dirs, "*.pkl"))

all_files.sort()
# for f in matching_files:
#     print(f)

output_folder = "visualized_slices"

if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

def auto_grid_size(n_images):
    """
    Compute rows and columns to make the grid as square as possible.
    Returns: (rows, cols)
    """
    cols = math.ceil(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)
    return rows, cols

def make_image_grid_auto(images, save_path):
    n_images = len(images)
    rows, cols = auto_grid_size(n_images)

    h, w, c = images[0].shape
    grid_img = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c_idx = idx % cols
        grid_img[r*h:(r+1)*h, c_idx*w:(c_idx+1)*w, :] = img

    Image.fromarray(grid_img).save(save_path)

# os.makedirs(output_folder, exist_ok=True)

images=[]
save_path=os.path.join(output_folder,f"{patient_id}.png")
for f in all_files:
    print(f)
    pkl_path = f
    with open(pkl_path, 'rb') as f:
         data = pickle.load(f)
    images.append(data)
make_image_grid_auto(images, save_path)


  
