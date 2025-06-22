import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()
ROOT=os.getenv("ROOT")

from model.Encoder.vit import ViT3DTower
def build_vision_tower(config, **kwargs):
# def build_vision_tower(tower, vision_select_layer,vision_select_feature,image_channel,image_size,patch_size, **kwargs):
    
    """
      VIT param
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    """
    vision_tower = getattr(config, 'vision_tower', None)

    if 'vit3d' in vision_tower.lower():
        return ViT3DTower(config,**kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')