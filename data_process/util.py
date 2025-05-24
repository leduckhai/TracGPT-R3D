
import numpy as np
import re
import pickle 
from sklearn.cluster import DBSCAN  # For spatial clustering

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import numpy as np

def rgb_to_grayscale(img_rgb):
    """Convert (H, W, 3) RGB to (H, W) grayscale using standard weights."""
    return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]) 


# def sort_files(file_list):
#     def sort_key(filename):
#         numbers = list(map(int, re.findall(r'\d+', filename)))[::-1]
#         return numbers 

#     sorted_files = sorted(file_list, key=sort_key)
#     return sorted_files

def sort_files(file_list):
    def sort_key(filename):
        numbers = list(map(int, re.findall(r'\d+', filename)))[::-1]
        return numbers 

    sorted_files = sorted(file_list, key=sort_key)
    return sorted_files

def sort_files_v2(file_list):
    return sorted(file_list)


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

def group_and_merge_3d_bboxes_v2(bboxes_with_z, 
                             img_size=None, eps=None, max_objects_k=None,
                             overlap_threshold=0.7, discard_inner_iou=0.9,min_slices=2):
    """
    3D bounding box merging with per-slice 2D IoU filtering
    
    Args:
        slice_to_bboxes: {slice_id: [bbox1, bbox2, ...]} (each bbox = [x_min,y_min,x_max,y_max])
        slice_positions: {slice_id: z_position}
        img_size: (width, height) for normalization
        eps: DBSCAN clustering threshold
        max_objects_k: Max expected 3D objects
        overlap_threshold: Discard if 2D IoU > threshold (0-1)
        min_slices: Minimum slices required for a valid 3D object
        
    Returns:
        List of merged 3D bboxes: [[x_min, y_min, z_min, x_max, y_max, z_max], ...]
    """
    # bboxes=[(x_min, y_min, x_max, y_max,slice_pos), ...]
    if not img_size:
        img_size=[1,1]
    width, height = img_size
    sorted_boxes = sorted(bboxes_with_z, 
                         key=lambda b: (b[2]-b[0])*(b[3]-b[1]), 
                         reverse=True)
    
    keep_boxes = {}
    for box in sorted_boxes:
        x1, y1, x2, y2, z = box
        box_2d = (x1, y1, x2, y2)
        discard = False
        found_match = False

        for kept_2d in list(keep_boxes.keys()):
            iou = calculate_2d_iou(box_2d, kept_2d)
            if iou > discard_inner_iou:
                discard = True
                break
            elif iou > overlap_threshold:
                keep_boxes[kept_2d][0] = min(keep_boxes[kept_2d][0], z)
                keep_boxes[kept_2d][1] = max(keep_boxes[kept_2d][1], z)
                found_match = True
                break

        if not discard and not found_match:
            keep_boxes[box_2d] = [z, z]

    # Post-process
    output = []
    for box_2d, (z_min, z_max) in keep_boxes.items():
        if (z_max - z_min + 1) >= min_slices:
            x_min, y_min, x_max, y_max = box_2d
            output.append((
                x_min * width, y_min * height, z_min,
                x_max * width, y_max * height, z_max
            ))

    return output

def merge_2d_bboxes_to_3d(list_of_2d_bboxes, slice_positions, img_size=None):
    """
    Merge multiple 2D bounding boxes from slices into a single 3D bounding box.
    
    Args:
        list_of_2d_bboxes: List of 2D bboxes in format [[x,y,w,h], ...]
        slice_positions: List of z-coordinates for each slice (same length as list_of_2d_bboxes)
        img_size: (width, height) tuple for normalization (optional)
    
    Returns:
        Single 3D bbox as [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    assert len(list_of_2d_bboxes) == len(slice_positions), \
           "Number of 2D boxes must match number of slice positions"
    
    all_corners_3d = []
    
    for bbox_2d, z in zip(list_of_2d_bboxes, slice_positions):
        # Unpack and optionally normalize
        x, y, w, h = bbox_2d
        if img_size:
            x, w = x * img_size[0], w * img_size[0]
            y, h = y * img_size[1], h * img_size[1]
        
        # Get 2D box corners
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        
        # Convert to 3D corners
        corners_3d = [
            [x_min, y_min, z],
            [x_min, y_max, z],
            [x_max, y_min, z],
            [x_max, y_max, z]
        ]
        all_corners_3d.extend(corners_3d)
    
    if not all_corners_3d:
        raise ValueError("No valid bounding boxes provided")
    
    # Compute min/max bounds
    all_corners_3d = np.array(all_corners_3d)
    bbox_min = np.min(all_corners_3d, axis=0)
    bbox_max = np.max(all_corners_3d, axis=0)
    
    return np.concatenate([bbox_min, bbox_max])

def group_and_merge_3d_bboxes(slice_to_bboxes, slice_positions, img_size=None, eps=None, max_objects_k=None):
    """
    Args:
        slice_to_bboxes: {slice_id: [bbox1, bbox2, ...]} (each bbox = [x_min,y_min,x_max,y_max])
        slice_positions: {slice_id: z_position}
        img_size: (width, height) for normalization (optional)
        eps: Clustering threshold (manual override)
        max_objects_k: Prior knowledge of max 3D objects (optional)
    Returns:
        List of merged 3D bboxes: [[x_min, y_min, z_min, x_max, y_max, z_max], ...]
    """
    # Step 1: Flatten all boxes with slice positions and normalize if needed
    all_boxes = []
    for slice_id, bboxes in slice_to_bboxes.items():
        z = slice_positions[slice_id]
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            
            # Normalize coordinates if image size is provided
            if img_size is not None:
                img_w, img_h = img_size
                x_min, x_max = x_min/img_w, x_max/img_w
                y_min, y_max = y_min/img_h, y_max/img_h
            
            center = np.array([(x_min + x_max)/2, (y_min + y_max)/2, z])
            all_boxes.append((center, [x_min, y_min, x_max, y_max, z]))
    
    if not all_boxes:
        return []

    centers = np.array([box[0] for box in all_boxes])

    # Step 2: Auto-estimate eps if not provided
    if eps is None:
        if len(centers) > 1:
            if max_objects_k is not None and max_objects_k > 0:
                # Adaptive threshold based on desired number of clusters
                pairwise_dists = pdist(centers)
                eps = np.percentile(pairwise_dists, 100 * (1 - 1/max_objects_k))
            else:
                # Default: use nearest neighbor distance
                neigh = NearestNeighbors(n_neighbors=2)
                neigh.fit(centers)
                distances, _ = neigh.kneighbors(centers)
                eps = np.median(distances[:, 1]) * 1.5
            print(f"Auto-selected eps: {eps:.3f}")
        else:
            eps = 0.1  # Default small value if only one box exists
    
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centers)  # min_samples=1 to allow single-box clusters
    labels = clustering.labels_
    
    # Step 4: Merge boxes per cluster
    bboxes_3d = []
    for label in set(labels):
        if label == -1:  # Skip noise points if any
            continue
            
        cluster_boxes = [all_boxes[i][1] for i in np.where(labels == label)[0]]
        
        # Get all corners (8 corners per box)
        corners_3d = []
        for x1, y1, x2, y2, z in cluster_boxes:
            corners_3d.append([x1, y1, z])
            corners_3d.append([x1, y2, z])
            corners_3d.append([x2, y1, z])
            corners_3d.append([x2, y2, z])
        
        corners_3d = np.array(corners_3d)
        
        # Create merged 3D bbox
        bboxes_3d.append([
            corners_3d[:, 0].min(),  # x_min
            corners_3d[:, 1].min(),  # y_min
            corners_3d[:, 2].min(),  # z_min
            corners_3d[:, 0].max(),  # x_max
            corners_3d[:, 1].max(),  # y_max
            corners_3d[:, 2].max()   # z_max
        ])
    
    # Step 5: Enforce max objects constraint if specified
    if max_objects_k is not None and len(bboxes_3d) > max_objects_k:
        # Sort by 3D volume (descending) and keep top-k
        bboxes_3d.sort(
            key=lambda b: (b[3]-b[0])*(b[4]-b[1])*(b[5]-b[2]), 
            reverse=True
        )
        bboxes_3d = bboxes_3d[:max_objects_k]
        print(f"Kept top {max_objects_k} largest 3D boxes")
    
    return bboxes_3d

def bboxes_overlap_2d(bbox1, bbox2):
    """
    Check if two 2D bounding boxes overlap (IoU > 0).
    Format: [x_min, y_min, x_max, y_max]
    """
    # Check overlap along x and y axes
    x_overlap = (bbox1[0] < bbox2[2]) and (bbox1[2] > bbox2[0])
    y_overlap = (bbox1[1] < bbox2[3]) and (bbox1[3] > bbox2[1])
    return x_overlap and y_overlap



def draw_3d_bbox_wireframe_v2(shape, bboxes, color=(0, 1, 0), line_thickness=1):
    """
    Create a 3D volume with green wireframe bounding boxes drawn.
    
    Args:
        shape: Tuple of (Z, H, W) dimensions for the output volume
        bboxes: List of bboxes in format [x_min, y_min, z_min, x_max, y_max, z_max]
                or a single bbox (in W,H,Z coordinates)
        color: RGB color tuple (default (0,1,0) for green)
        line_thickness: Number of voxels for line width (1 = single voxel line)
    
    Returns:
        3D numpy array (Z, H, W, 3) with colored wireframes (black background)
    """
    # Initialize RGB volume (black background)
    volume = np.zeros(shape + (3,), dtype=np.float32)
    
    # Convert single bbox to list for uniform processing
    if isinstance(bboxes, (list, np.ndarray)) and len(bboxes) == 6 and not isinstance(bboxes[0], (list, np.ndarray)):
        bboxes = [bboxes]
    
    for bbox in bboxes:
        # Extract and round coordinates (x=W, y=H, z=Z)
        x_min, y_min, z_min, x_max, y_max, z_max = [int(round(c)) for c in bbox]
        
        # Clip coordinates to volume dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)
        x_max = min(shape[2]-1, x_max)
        y_max = min(shape[1]-1, y_max)
        z_max = min(shape[0]-1, z_max)
        
        # Calculate thickness ranges
        t = line_thickness
        t_range = range(-(t//2), t//2 + 1)
        
        # Draw all 12 edges of the bounding box in green
        # Horizontal edges (along x/W axis)
        for dt in t_range:
            # Bottom edges
            volume[z_min, y_min:y_min+t, x_min:x_max+1, :] = color  # Front
            volume[z_min, y_max-t+1:y_max+1, x_min:x_max+1, :] = color  # Back
            # Top edges
            volume[z_max, y_min:y_min+t, x_min:x_max+1, :] = color  # Front
            volume[z_max, y_max-t+1:y_max+1, x_min:x_max+1, :] = color  # Back
            
        # Vertical edges (along y/H axis)
        for dt in t_range:
            # Left edges
            volume[z_min:z_max+1, y_min:y_min+t, x_min, :] = color  # Front
            volume[z_min:z_max+1, y_min:y_min+t, x_max, :] = color  # Back
            # Right edges
            volume[z_min:z_max+1, y_max-t+1:y_max+1, x_min, :] = color  # Front
            volume[z_min:z_max+1, y_max-t+1:y_max+1, x_max, :] = color  # Back
            
        # Depth edges (along z/Z axis)
        for dt in t_range:
            # Front edges
            volume[z_min:z_max+1, y_min, x_min:x_min+t, :] = color  # Bottom
            volume[z_min:z_max+1, y_min, x_max-t+1:x_max+1, :] = color  # Top
            # Back edges
            volume[z_min:z_max+1, y_max, x_min:x_min+t, :] = color  # Bottom
            volume[z_min:z_max+1, y_max, x_max-t+1:x_max+1, :] = color  # Top

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
        # convert_slice_path_to_3d(slice_path)
    # return [convert_slice_path_to_3d(slice_path) for slice_path in list_slice_paths]

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
    # Convert RGB to grayscale for saving (e.g., mean of channels)
        sample_to_save = sample.mean(axis=-1)  # (Z, H, W)
    else:
        sample_to_save = sample  # Already suitable shape
    print("sample_to_save",sample_to_save.shape)
    nifti_img = nib.Nifti1Image(sample_to_save, affine=np.eye(4))
    nib.save(nifti_img, "save.nii.gz")

