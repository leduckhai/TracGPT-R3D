import os
import re
import pickle
import numpy as np 
import nibabel as nib
folder_path = "/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/image/OAS1_0002"

def sort_files(file_list):
    def sort_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))[::-1]
        return numbers 

    sorted_files = sorted(file_list, key=sort_key)
    return sorted_files

files=os.listdir(folder_path)
files=sort_files(files)
stack=[]
for file in files:
        with open(os.path.join(folder_path, file), 'rb') as f:
            data = pickle.load(f)
            data=np.mean(data, axis=2)
        stack.append(data)
stack=np.stack(stack,axis=0)
affine = np.eye(4)

nifti_img = nib.Nifti1Image(stack, affine)
if os.path.exists(f"file_concat.nii"):
    os.remove(f"file_concat.nii")
nib.save(nifti_img, f"file_concat.nii")

