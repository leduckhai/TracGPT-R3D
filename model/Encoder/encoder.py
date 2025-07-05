import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()
ROOT=os.getenv("ROOT")
import sys
sys.path.append(ROOT)
from types import SimpleNamespace
from model.Encoder.vit import ViT3DTower

def build_vision_tower(config):
  
    if 'vit3d' in config.vision_tower.lower():
        return ViT3DTower(
            in_channels=config.in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            vision_select_layer=config.vision_select_layer,
            vision_select_feature=config.vision_select_feature
        )
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
if __name__ == "__main__":
    vision_tower = build_vision_tower('vit3d')
    print(vision_tower)
    print(vision_tower.hidden_size)
    print(vision_tower.device)
    print(vision_tower.dtype)