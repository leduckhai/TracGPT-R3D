import numpy as np
import nibabel as nib
import pickle
import os
from util import convert_list_slice_paths_to_3d

pickle_base="/home/ducnguyen/sync_local/repo/TracGPT/clean_data_3d_nodup/c63028d8-d000-44e5-875a-da9d453bbbf0/train/image_with_bboxes"

patient_ids=["OAS1_0002","OAS1_0003","OAS1_0004","OAS1_0005"]


def save_nifti(array_3d, output_path, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Save a 3D numpy array as a NIfTI file.
    
    Parameters:
        array_3d: np.ndarray
            3D image data with shape (Z, Y, X)
        output_path: str
            Output path ending in .nii or .nii.gz
        voxel_spacing: tuple of 3 floats
            Size of each voxel in mm (default: isotropic 1mm)
    """
    if array_3d.ndim != 3:
        raise ValueError("Expected a 3D array (Z, Y, X)")

    affine = np.diag(voxel_spacing + (1,))  # 4x4 identity matrix with spacing
    nifti_img = nib.Nifti1Image(array_3d, affine)
    nib.save(nifti_img, output_path)
    print(f"NIfTI saved to: {output_path}")



# for patient_id in patient_ids:
#     patient_path = os.path.join(pickle_base, f"{patient_id}.pkl")
#     with open(patient_path, "rb") as f:
#         data = pickle.load(f)
#         print("df",data.shape)
#     output_path=f"{patient_id}.nii.gz"
#     if os.path.exists(output_path):
#         os.remove(output_path)
#     save_nifti(data, output_path, voxel_spacing=(1.0, 1.0, 1.0))
# Example usage:
# arr = np.random.rand(128, 256, 256).astype(np.float32)
# save_nifti(arr, "output_image.nii.gz")
if __name__=="__main__":
    import re
    base_dir="/home/ducnguyen/sync_local/repo/TracGPT/clean_data/train/image_with_bboxes/OAS1_0002"
    file_pkls = [f for f in os.listdir(base_dir) if f.endswith('.pkl')]
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]
    # sorted(file_pkls)
    sorted_files = sorted(file_pkls, key=natural_sort_key)
    full_path=[os.path.join(base_dir,f) for f in sorted_files]
    print("file_pkls",sorted_files)
    img_3d=convert_list_slice_paths_to_3d(full_path)
    print("img_3d",img_3d.shape)
    output_path=f"OAS1_0002.nii.gz"
    if os.path.exists(output_path):
        os.remove(output_path)
    save_nifti(img_3d, output_path, voxel_spacing=(1.0, 1.0, 1.0))