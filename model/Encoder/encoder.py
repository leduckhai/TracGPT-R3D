import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()
ROOT=os.getenv("ROOT")

from model.vit import ViT3DTower
def build_vision_tower(config, **kwargs):
# def build_vision_tower(tower, vision_select_layer,vision_select_feature,image_channel,image_size,patch_size, **kwargs):

    vision_tower = getattr(config, 'vision_tower', None)

    if 'vit3d' in vision_tower.lower():
        return ViT3DTower(config,**kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')