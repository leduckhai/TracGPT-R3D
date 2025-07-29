import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from glob import glob
from scipy.ndimage import zoom
import sys 
sys.path.append("/root/TracGPT-R3D")
import os
import json
import nibabel as nib
def load_mhd_file(mhd_path):
    """Load an MHD/ZRAW image pair with proper metadata handling"""
    image = sitk.ReadImage(mhd_path)
    
    # Get crucial metadata for coordinate conversion
    spacing = np.array(image.GetSpacing())  # (x,y,z) spacing in mm
    origin = np.array(image.GetOrigin())    # (x,y,z) origin in mm
    
    # Convert to numpy array (Z,Y,X order)
    array = sitk.GetArrayFromImage(image)  
    
    return {
        'image': array,
        'spacing': spacing,
        'origin': origin,
        'direction': image.GetDirection()  # Image orientation
    }

def load_luna16_annotations(annotations_csv, image_metadata):
    """Load annotations with proper world-to-voxel conversion"""
    df = pd.read_csv(annotations_csv)
    bboxes = []
    
    for _, row in df.iterrows():
        # Convert world coordinates (mm) to voxel coordinates
        world_coords = np.array([row.coordX, row.coordY, row.coordZ])
        
        # Proper conversion using image metadata
        voxel_coords = (world_coords - image_metadata['origin']) / image_metadata['spacing']
        
        # Convert diameter from mm to voxels
        diameter_voxels = row.diameter_mm / image_metadata['spacing']
        
        # Center-based format: (cx, cy, cz, dx, dy, dz) in voxel coordinates
        bbox = np.concatenate([voxel_coords, diameter_voxels])
        bboxes.append(bbox)
    
    return np.array(bboxes)

class Luna16Dataset(Dataset):
    def __init__(self, data_dir, annotations_csv, target_shape=(128, 128, 128)):
        self.mhd_files = sorted(glob(f"{data_dir}/*.mhd"))
        self.annotations_csv = annotations_csv
        self.target_shape = target_shape
        
    def __len__(self):
        return len(self.mhd_files)
    
    def __getitem__(self, idx):
        # Load image with metadata
        mhd_path = self.mhd_files[idx]
        data = load_mhd_file(mhd_path)
        ct_volume = data['image']
        
        # Load annotations with proper coordinate conversion
        bboxes = load_luna16_annotations(self.annotations_csv, {
            'spacing': data['spacing'],
            'origin': data['origin']
        })
        
        # Normalize to [-1, 1] range (common for CT scans)
        ct_volume = np.clip(ct_volume, -1000, 400)  # Typical lung window
        ct_volume = (ct_volume - (-1000)) / (400 - (-1000)) * 2 - 1  # Scale to [-1, 1]
        
        # Resample to target shape
        zoom_factors = np.array(self.target_shape) / np.array(ct_volume.shape)
        ct_volume = zoom(ct_volume, zoom_factors, order=3)  # Cubic interpolation
        
        # Adjust bboxes for resizing
        bboxes = bboxes.copy()
        for i in range(3):  # Adjust center coordinates
            bboxes[:, i] *= zoom_factors[2-i]  # ZYX order conversion
        for i in range(3, 6):  # Adjust diameters
            bboxes[:, i] *= zoom_factors[5-i]
            
        return {
            'image': torch.tensor(ct_volume, dtype=torch.float32).unsqueeze(0),
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'original_spacing': torch.tensor(data['spacing']),
            'file_path': mhd_path
        }

def save_as_nii(image_tensor, bbox_tensor, original_spacing, output_dir, filename):
    """
    Save 3D image and bboxes as NIfTI files
    
    Args:
        image_tensor: (1, D, H, W) tensor
        bbox_tensor: (N, 6) tensor of (cx,cy,cz,dx,dy,dz)
        original_spacing: (3,) original spacing in mm (as tensor or numpy array)
        output_dir: Directory to save files
        filename: Base filename (without extension)
    """
    import os
    import json
    import nibabel as nib
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert all inputs to numpy
    image_np = image_tensor.squeeze(0).cpu().numpy()  # (D, H, W)
    bboxes_np = bbox_tensor.cpu().numpy() if torch.is_tensor(bbox_tensor) else bbox_tensor  # (N, 6)
    spacing_np = original_spacing.cpu().numpy() if torch.is_tensor(original_spacing) else np.array(original_spacing)
    
    # Calculate new spacing after any resizing
    current_shape = np.array(image_np.shape)
    original_shape = current_shape * (spacing_np / spacing_np.mean())
    new_spacing = spacing_np * (original_shape / current_shape)
    
    # Create affine matrix (using RAS+ orientation)
    affine = np.eye(4)
    affine[:3, :3] = np.diag(new_spacing)
    
    # Save image
    img_nii = nib.Nifti1Image(image_np, affine)
    nib.save(img_nii, os.path.join(output_dir, f"{filename}_image.nii.gz"))
    
    # Create and save bbox mask
    if len(bboxes_np) > 0:
        bbox_mask = np.zeros_like(image_np, dtype=np.uint8)
        
        for bbox in bboxes_np:
            cx, cy, cz, dx, dy, dz = bbox
            # Convert to voxel coordinates
            z_start = max(0, int(np.round(cz - dz/2)))
            z_end = min(image_np.shape[0], int(np.round(cz + dz/2)))
            y_start = max(0, int(np.round(cy - dy/2)))
            y_end = min(image_np.shape[1], int(np.round(cy + dy/2)))
            x_start = max(0, int(np.round(cx - dx/2)))
            x_end = min(image_np.shape[2], int(np.round(cx + dx/2)))
            
            if z_start < z_end and y_start < y_end and x_start < x_end:
                bbox_mask[z_start:z_end, y_start:y_end, x_start:x_end] = 1
        
        mask_nii = nib.Nifti1Image(bbox_mask, affine)
        nib.save(mask_nii, os.path.join(output_dir, f"{filename}_bboxes.nii.gz"))
    
    metadata = {
        'original_spacing': spacing_np.tolist(),
        'bboxes': bboxes_np.tolist(),
        'current_spacing': new_spacing.tolist(),
        'original_shape': current_shape.tolist()
    }
    with open(os.path.join(output_dir, f"{filename}_meta.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
        
# Example usage:
if __name__ == "__main__":
    dataset = Luna16Dataset(
        data_dir="/root/TracGPT-R3D/DATASET/luna/seg-lungs-LUNA16/seg-lungs-LUNA16",
        annotations_csv="/root/TracGPT-R3D/DATASET/luna/annotations.csv"
    )
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Bounding boxes: {sample['bboxes']}")
    print(f"Original spacing (mm): {sample['original_spacing']}")
    save_as_nii(sample['image'],sample['bboxes'],sample['original_spacing'], ".", "lunasample")