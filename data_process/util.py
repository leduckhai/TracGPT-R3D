
import numpy as np
import re
import pickle 
from sklearn.cluster import DBSCAN  # For spatial clustering

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import numpy as np
import nibabel as nib
def rgb_to_grayscale(img_rgb):
    """Convert (H, W, 3) RGB to (H, W) grayscale using standard weights."""
    return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]) 



def sort_files(file_list):
    def sort_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))[::-1]
        return numbers 

    sorted_files = sorted(file_list, key=sort_key)
    return sorted_files


def calculate_2d_iou(box1, box2):
    """Calculate 2D IoU between two bounding boxes on the same slice"""
    # Box format: [x_min, y_min, x_max, y_max]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    
    return inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0


def calculate_2d_iou_of_multiple_boxes(boxes1, boxes2):
    ious=[]
    for box1 in boxes1:
        max_iou=0
        for box2 in boxes2:
            iou = calculate_2d_iou(box1, box2)
            max_iou=max(max_iou,iou)
        ious.append(max_iou)
    return np.array(ious).mean()

def group_and_merge_3d_bboxes_v2(bboxes_slice, num_concat=50, 
                                 img_size=None, eps=None, max_objects_k=None,
                                 overlap_threshold=0.8, discard_inner_iou=0.9,
                                 min_slices=2, continuity_threshold=0.7):

    if not bboxes_slice:
        return []

    width, height = img_size if img_size else (1, 1)

    # Group slice indices with similar boxes
    slice_groups = []

    for i, current_boxes in enumerate(bboxes_slice):
        if not current_boxes:
            continue

        matched = False

        for group in slice_groups:
            reference_slice_idx = group[-1]
            reference_boxes = bboxes_slice[reference_slice_idx]

            if calculate_2d_iou_of_multiple_boxes(current_boxes, reference_boxes) > overlap_threshold:
                group.append(i)
                matched = True
                break

        if not matched:
            slice_groups.append([i])

    output = []
    for group_indices in slice_groups:
        if len(group_indices) < min_slices:
            continue

        z_min = min(group_indices) / num_concat
        z_max = max(group_indices) / num_concat

        group_continuity = len(group_indices) / (max(group_indices) - min(group_indices) + 1)
        if group_continuity < continuity_threshold:
            print("skip non_continuous group", group_continuity)
            continue

        for bbox in bboxes_slice[group_indices[0]]:
            x_min, y_min, x_max, y_max = bbox
            output.append([x_min, y_min, z_min, x_max, y_max, z_max])

    return output
  
def bboxes_overlap_2d(bbox1, bbox2):
    """
    Check if two 2D bounding boxes overlap (IoU > 0).
    Format: [x_min, y_min, x_max, y_max]
    """
    # Check overlap along x and y axes
    x_overlap = (bbox1[0] < bbox2[2]) and (bbox1[2] > bbox2[0])
    y_overlap = (bbox1[1] < bbox2[3]) and (bbox1[3] > bbox2[1])
    return x_overlap and y_overlap

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



def draw_3d_bbox_wireframe_v2(shape, bboxes, line_thickness=1):
    """
    Create a 3D binary volume with wireframe bounding boxes.

    Args:
        shape: Tuple of (Z, H, W) dimensions for the output volume
        bboxes: List of bboxes in format [x_min, y_min, z_min, x_max, y_max, z_max]
                or a single bbox
        line_thickness: Number of voxels for line width (1 = single voxel line)

    Returns:
        3D numpy array (Z, H, W) with wireframe lines drawn as 1s (0 background)
    """
    # Initialize volume (single channel)
    volume = np.zeros(shape, dtype=np.uint8)

    # Handle single bbox input
    if isinstance(bboxes, (list, np.ndarray)) and len(bboxes) == 6 and not isinstance(bboxes[0], (list, np.ndarray)):
        bboxes = [bboxes]

    for bbox in bboxes:
        x_min, y_min, z_min, x_max, y_max, z_max = [int(round(c)) for c in bbox]

        # Clip to bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)
        x_max = min(shape[2]-1, x_max)
        y_max = min(shape[1]-1, y_max)
        z_max = min(shape[0]-1, z_max)

        t = line_thickness
        t_range = range(-(t//2), t//2 + 1)

        # Horizontal edges (X axis)
        volume[z_min, y_min:y_min+t, x_min:x_max+1] = 1  # Bottom Front
        volume[z_min, y_max-t+1:y_max+1, x_min:x_max+1] = 1  # Bottom Back
        volume[z_max, y_min:y_min+t, x_min:x_max+1] = 1  # Top Front
        volume[z_max, y_max-t+1:y_max+1, x_min:x_max+1] = 1  # Top Back

        # Vertical edges (Y axis)
        volume[z_min:z_max+1, y_min:y_min+t, x_min] = 1  # Left Front
        volume[z_min:z_max+1, y_min:y_min+t, x_max] = 1  # Right Front
        volume[z_min:z_max+1, y_max-t+1:y_max+1, x_min] = 1  # Left Back
        volume[z_min:z_max+1, y_max-t+1:y_max+1, x_max] = 1  # Right Back

        # Depth edges (Z axis)
        volume[z_min:z_max+1, y_min, x_min:x_min+t] = 1  # Front Bottom
        volume[z_min:z_max+1, y_min, x_max-t+1:x_max+1] = 1  # Front Top
        volume[z_min:z_max+1, y_max, x_min:x_min+t] = 1  # Back Bottom
        volume[z_min:z_max+1, y_max, x_max-t+1:x_max+1] = 1  # Back Top

    return volume

def draw_3d_bbox_filled(shape, bboxes):
    """
    Create a 3D volume with filled bounding boxes, each having a unique label.

    Args:
        shape: Tuple of (Z, H, W) dimensions for the output volume.
        bboxes: List of bboxes in format [x_min, y_min, z_min, x_max, y_max, z_max]
                or a single bbox.

    Returns:
        3D numpy array (Z, H, W) with filled boxes labeled with unique integers (1, 2, 3, ...).
        Background is 0.
    """
    volume = np.zeros(shape, dtype=np.uint16)  # Use uint16 to support many labels

    # Handle single bbox input
    if isinstance(bboxes, (list, np.ndarray)) and len(bboxes) == 6 and not isinstance(bboxes[0], (list, np.ndarray)):
        bboxes = [bboxes]

    for idx, bbox in enumerate(bboxes, start=1):  # Start labels from 1
        x_min, y_min, z_min, x_max, y_max, z_max = [int(round(c)) for c in bbox]

        # Clip to volume bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)
        x_max = min(shape[2] - 1, x_max)
        y_max = min(shape[1] - 1, y_max)
        z_max = min(shape[0] - 1, z_max)

        volume[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = idx

    return volume


def draw_3d_bbox_labels(shape, bboxes):
    """
    Create a label volume with unique values per bounding box.

    Args:
        shape: Tuple (D, H, W) – the shape of the output volume.
        bboxes: List of [x_min, y_min, z_min, x_max, y_max, z_max] boxes.

    Returns:
        A 3D numpy array with labels.
    """
    volume = np.zeros(shape, dtype=np.uint8)

    for i, bbox in enumerate(bboxes, start=1):
        x_min, y_min, z_min, x_max, y_max, z_max = [int(round(c)) for c in bbox]

        # Clip to bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)
        x_max = min(shape[2] - 1, x_max)
        y_max = min(shape[1] - 1, y_max)
        z_max = min(shape[0] - 1, z_max)

        volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = i

    return volume


def draw_3d_bbox_wireframe(volume, bboxes, color=1.0, clip_to_volume=True, line_thickness=3):
    """
    Draw wireframes for multiple 3D bounding boxes on a volume with adjustable line thickness.
    
    Args:
        volume: 3D numpy array (Z, Y, X).
        bboxes: List of bboxes in format [x_min, y_min, z_min, x_max, y_max, z_max],
                or a single bbox.
        color: Intensity value for box edges.
        clip_to_volume: Whether to clip boxes to volume dimensions.
        line_thickness: Number of voxels to expand each edge (1 = single voxel line)
    
    Returns:
        Volume with wireframes drawn (modified in-place).
    """
    # Convert single bbox to list for uniform processing
    if isinstance(bboxes, (list, np.ndarray)) and len(bboxes) == 6:
        bboxes = [bboxes]
    
    for bbox in bboxes:
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        
        if clip_to_volume:
            # Clip coordinates to volume dimensions with thickness padding
            x_min = max(0, int(x_min) - line_thickness)
            y_min = max(0, int(y_min) - line_thickness)
            z_min = max(0, int(z_min) - line_thickness)
            x_max = min(volume.shape[2]-1, int(x_max) + line_thickness)
            y_max = min(volume.shape[1]-1, int(y_max) + line_thickness)
            z_max = min(volume.shape[0]-1, int(z_max) + line_thickness)
        else:
            # Convert to integers with thickness expansion
            x_min, y_min, z_min = int(x_min)-line_thickness, int(y_min)-line_thickness, int(z_min)-line_thickness
            x_max, y_max, z_max = int(x_max)+line_thickness, int(y_max)+line_thickness, int(z_max)+line_thickness

        # ---- X-aligned edges (vertical pillars) ----
        # Expanded using slice ranges
        for offset in range(line_thickness+1):
            # Left pillars
            volume[z_min:z_max+1, 
                  y_min+offset, 
                  x_min+offset] = color
            volume[z_min:z_max+1, 
                  y_max-offset, 
                  x_min+offset] = color
            # Right pillars
            volume[z_min:z_max+1, 
                  y_min+offset, 
                  x_max-offset] = color
            volume[z_min:z_max+1, 
                  y_max-offset, 
                  x_max-offset] = color

        # ---- Y-aligned edges (top/bottom frames) ----
        for offset in range(line_thickness+1):
            # Bottom frame
            volume[z_min+offset, 
                  y_min:y_max+1, 
                  x_min+offset] = color
            volume[z_min+offset, 
                  y_min:y_max+1, 
                  x_max-offset] = color
            # Top frame
            volume[z_max-offset, 
                  y_min:y_max+1, 
                  x_min+offset] = color
            volume[z_max-offset, 
                  y_min:y_max+1, 
                  x_max-offset] = color

        # ---- Z-aligned edges (side connectors) ----
        for offset in range(line_thickness+1):
            # Front connectors
            volume[z_min+offset, 
                  y_min+offset, 
                  x_min:x_max+1] = color
            volume[z_min+offset, 
                  y_max-offset, 
                  x_min:x_max+1] = color
            # Back connectors
            volume[z_max-offset, 
                  y_min+offset, 
                  x_min:x_max+1] = color
            volume[z_max-offset, 
                  y_max-offset, 
                  x_min:x_max+1] = color

    return volume

def combine_to_3d_bbox(boxes_2d, slice_positions, img_width=None, img_height=None):
    """
    Combine 2D bounding boxes from multiple slices into a 3D bounding box.

    Args:
        boxes_2d: List of 2D boxes in format [[x, y, w, h], ...] (normalized or pixel coords).
        slice_positions: List of Z-coordinates for each slice (e.g., [z1, z2, ...]).
        img_width, img_height: Required if boxes are normalized (to unnormalize).
    
    Returns:
        3D bounding box as [x_min, y_min, z_min, x_max, y_max, z_max].
    """
    if len(boxes_2d) != len(slice_positions):
        raise ValueError("boxes_2d and slice_positions must have the same length.")

    all_corners_3d = []
    for (x, y, w, h), z in zip(boxes_2d, slice_positions):
        if None in (x, y, w, h, z):
            raise ValueError("All 2D box coordinates and Z must be provided.")
        # If normalized, convert to pixel coordinates
        if img_width and img_height:
            x = x * img_width
            y = y * img_height
            w = w * img_width
            h = h * img_height

        # Get 2D box corners
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h

        # Convert to 3D by adding Z-coordinate
        corners_3d = [
            [x_min, y_min, z],
            [x_min, y_max, z],
            [x_max, y_min, z],
            [x_max, y_max, z]
        ]
        all_corners_3d.extend(corners_3d)
        # print("type", type(corners_3d),type(corners_3d[0][0]))
        # print("corners 3d", corners_3d)

    # Compute min/max bounds to form the 3D box
    all_corners_3d = np.array(all_corners_3d)
    bbox_3d_min = np.min(all_corners_3d, axis=0)
    bbox_3d_max = np.max(all_corners_3d, axis=0)

    return np.concatenate([bbox_3d_min, bbox_3d_max])  # [x_min, y_min, z_min, x_max, y_max, z_max]


def convert_list_slice_paths_to_3d(list_slice_paths):
    slice_stack=[]
    for slice_path in list_slice_paths:
        if slice_path.split("/")[-1].split(".")[1]=="pkl":
            with open(slice_path, "rb") as f:
                # slice_stack.append(pickle.load(f))
                slice=pickle.load(f)
                if slice.shape[-1]==3:
                    # slice_stack=np.concatenate(slice_stack, axis=2)
                    slice=rgb_to_grayscale(slice)
                slice_stack.append(slice)
        else:
            raise NotImplementedError
    return np.stack(slice_stack, axis=0)

def show_rgb_slices(data, num_slices=5, figsize=(15, 5), normalize=True, save_path=None, show=True):
    """
    Visualize and optionally save slices from a 4D array of shape (Z, H, W, 3).

    Parameters:
    - data (np.ndarray): Input array of shape (Z, H, W, 3).
    - num_slices (int): Number of slices to display.
    - figsize (tuple): Figure size for matplotlib.
    - normalize (bool): Whether to normalize data to [0, 1] for display.
    - save_path (str): Path to save the image (e.g., 'output/slices.png'). If None, does not save.
    - show (bool): Whether to display the figure using plt.show().
    """
    assert data.ndim == 4 and data.shape[-1] == 3, "Input must be of shape (Z, H, W, 3)"
    
    Z = data.shape[0]
    indices = np.linspace(0, Z - 1, num_slices, dtype=int)

    fig, axs = plt.subplots(1, num_slices, figsize=figsize)
    for i, idx in enumerate(indices):
        img = data[idx]
        if normalize:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axs[i].imshow(img)
        axs[i].set_title(f"Slice {idx}")
        axs[i].axis('off')
    
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    pass
def draw_bbox_volume(volume, bboxes, color=(1, 0, 0), thickness=2):
    """
    Draws 3D bounding boxes (projected per-slice) on a volume.

    Parameters:
    - volume (np.ndarray): Shape (Z, H, W, 3).
    - bboxes (list of list or np.ndarray): Each bbox is [x_min, y_min, z_min, x_max, y_max, z_max).
    - color (tuple): RGB in range [0, 1].
    - thickness (int): Border thickness in pixels.

    Returns:
    - np.ndarray: Volume with bounding boxes drawn.
    """
    import numpy as np

    assert volume.ndim == 4 and volume.shape[-1] == 3, "Volume must have shape (Z, H, W, 3)"

    vol = volume.copy()

    for bbox in bboxes:
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        for z in range(z_min, z_max):
            for t in range(thickness):
                # Top edge
                if y_min + t < vol.shape[1]:
                    vol[z, y_min + t, x_min:x_max] = color
                # Bottom edge
                if y_max - 1 - t >= 0:
                    vol[z, y_max - 1 - t, x_min:x_max] = color
                # Left edge
                if x_min + t < vol.shape[2]:
                    vol[z, y_min:y_max, x_min + t] = color
                # Right edge
                if x_max - 1 - t >= 0:
                    vol[z, y_min:y_max, x_max - 1 - t] = color

    return vol

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import nibabel as nib
    # bboxes=[(140.3774601908569, 180.138030352515, 3, 224.89944293599297, 298.4104251387325, 19), (14.093393810843528, 192.54437709649008, 3, 98.88426695121106, 308.8217794877045, 19), (11.284689215375012, 268.66907757519834, 0, 88.57548113522245, 341.2333583241721, 25), (146.9672824584949, 193.77289367573056, 0, 225.2195865233837, 265.27079233070526, 25), (157.22403021917663, 283.83786688460214, 0, 234.7810404401599, 355.55463464445535, 25), (21.268113834960467, 205.22473875229565, 0, 98.89200036291736, 272.67317282613067, 25), (67.93044412772954, 388.32932598507807, 3, 160.21829561845595, 439.73223990742383, 19), (56.4229802230619, 303.21241091292035, 3, 118.80623380067001, 354.49541166892254, 19)]
    # part0
    # bboxes=[(148.40849888821168, 193.5152781694207, 0, 225.1370103483504, 267.6268004585002, 49), (158.01559406557539, 284.116611715099, 0, 235.33437658351212, 356.9941984487808, 49), (12.585243281510385, 267.89056429463625, 0, 88.62149447301381, 341.78197567838134, 48), (21.707243887701253, 203.86785194295098, 0, 97.67827229789175, 272.7820635871126, 47)]
    # part 200
    bboxes=[[(140.3774601908569, 180.138030352515, 7, 224.89944293599297, 298.4104251387325, 15), (14.093393810843528, 192.54437709649008, 7, 98.88426695121106, 308.8217794877045, 19), (11.284689215375012, 268.66907757519834, 0, 88.57548113522245, 341.2333583241721, 24), (146.9672824584949, 193.77289367573056, 1, 225.2195865233837, 265.27079233070526, 25), (157.22403021917663, 283.83786688460214, 0, 234.7810404401599, 355.55463464445535, 24), (21.268113834960467, 205.22473875229565, 0, 98.89200036291736, 272.67317282613067, 25), (67.93044412772954, 388.32932598507807, 7, 160.21829561845595, 439.73223990742383, 19), (56.4229802230619, 303.21241091292035, 3, 118.80623380067001, 354.49541166892254, 19)]]
    int_bboxes=[]
    for bbox in bboxes:
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        int_bboxes.append([int(x_min),int(y_min),int(z_min),int(x_max),int(y_max),int(z_max)])
    shape=(50, 496, 248,3)
    # img_3d=draw_3d_bbox_wireframe_v2(shape, bboxes)
    volume = np.zeros(shape, dtype=np.float32)
    sample=draw_bbox_volume(volume, int_bboxes)
    save_path="save.nii.gz"
    if sample.shape[-1] == 3:
        sample_to_save = sample.mean(axis=-1)  # (Z, H, W)
    else:
        sample_to_save = sample 
    print("sample_to_save",sample_to_save.shape)
    nifti_img = nib.Nifti1Image(sample_to_save, affine=np.eye(4))
    nib.save(nifti_img, "save.nii.gz")

