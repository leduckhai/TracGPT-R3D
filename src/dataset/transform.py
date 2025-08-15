import numpy as np
from monai.transforms import MapTransform, Compose, CropForeground, Resize
from typing import Mapping, Hashable
from monai.transforms.utils import map_binary_to_indices
from monai.transforms import Compose, ResizeD, EnsureChannelFirstD, SqueezeDimD
from monai.transforms import ScaleIntensityRanged
import monai.transforms as mtf
import torch

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


class ResizeBboxAndImage(MapTransform):
    def __init__(self, keys,  target_size=[32, 256, 256]):
        super().__init__(keys)
        self.target_size = np.array(target_size, dtype=np.float32)
        self.scale = self.target_size  

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        image=d["image"]
        image_shape=image.shape
        scale = self.target_size / image.shape

        return d


base_transform_3d=Compose(
            [
                EnsureChannelFirstD(keys=["image"], channel_dim="no_channel"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=255,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ResizeD(
                    keys=["image"],
                    spatial_size=[32, 256, 256],
                    mode="trilinear",
                    size_mode="all",
                ),
            ]
        )
base_transform_2d=Compose(
    [
        EnsureChannelFirstD(keys=["image"], channel_dim="no_channel"), 
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ResizeD(
            keys=["image"],
            spatial_size=[-1,256, 256], 
            # mode="bilinear",  
            mode="trilinear",
            size_mode="all",
        ),
    ]
)
train_transform = mtf.Compose(
        [
            mtf.RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
            mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
            mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            mtf.ToTensord(keys=["image"], dtype=torch.float),
        ]
    )

val_transform = mtf.Compose(
    [
        mtf.ToTensord(keys=["image"], dtype=torch.float),
    ]
)
