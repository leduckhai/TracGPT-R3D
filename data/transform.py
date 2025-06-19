import numpy as np
from monai.transforms import MapTransform, Compose, CropForeground, Resize
from typing import Mapping, Hashable
from monai.transforms.utils import map_binary_to_indices
class TrackCrop(MapTransform):
    """
    Apply CropForeground and record the cropping origin to adjust bboxes.
    """
    def __init__(self, image_key="image"):
        super().__init__(keys=image_key)
        self.image_key = image_key

    def __call__(self, data):
       
        d = dict(data)
        img = d[self.image_key]
        # Assume foreground is non-zero
        indices = map_binary_to_indices(img > 0)
        min_idx = np.min(indices, axis=0)
        max_idx = np.max(indices, axis=0) + 1
        d["crop_origin"] = min_idx
        d[self.image_key] = img[
            min_idx[0]:max_idx[0],
            min_idx[1]:max_idx[1],
            min_idx[2]:max_idx[2]
        ]
        d["cropped_size"] = d[self.image_key].shape
        return d

class NormalizeBBox3D(MapTransform):
    """
    Adjust bboxes after cropping and resizing.
    """
    def __init__(self, keys, crop_origin_key="crop_origin", cropped_size_key="cropped_size", target_size=[32, 256, 256]):
        super().__init__(keys)
        self.crop_origin_key = crop_origin_key
        self.cropped_size_key = cropped_size_key
        self.target_size = np.array(target_size)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        crop_origin = np.array(d[self.crop_origin_key])
        cropped_size = np.array(d[self.cropped_size_key])
        scale = self.target_size / cropped_size

        for key in self.keys:
            bbox = np.array(d[key])  # [z, y, x, dz, dy, dx]
            bbox[:3] = (bbox[:3] - crop_origin) * scale  # adjust position
            bbox[3:] = bbox[3:] * scale  # scale size
            d[key] = bbox
        return d

class ResizeBBox3D(MapTransform):
    def __init__(self, keys, orig_size, target_size=[32, 256, 256]):
        super().__init__(keys)
        self.orig_size = np.array(orig_size, dtype=np.float32)
        self.target_size = np.array(target_size, dtype=np.float32)
        self.scale = self.target_size  # [D, H, W]

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key in self.keys:
            bboxes=[]
            for bbox in d[key]:
                bbox = np.array(bbox, dtype=np.float32)
                bbox[:3] = bbox[:3] * self.scale
                bbox[3:] = bbox[3:] * self.scale
                bboxes.append(bbox)
            d[key] = np.array(bboxes, dtype=np.float32)
          
        return d