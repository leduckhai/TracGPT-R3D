import sys 
sys.path.append("/root/TracGPT-R3D")
from src.model.vision_encoder.vit import ViT3DTower

def load_vision_encoder(config,pretrained_path=None):
    if config["vision_tower"]=="vit3d":
        in_channels = config["in_channels"]
        img_size = config["img_size"]
        patch_size = config["patch_size"]
        hidden_size = config["hidden_size"]
        num_heads = config["num_heads"]
        vision_select_layer = config["vision_select_layer"]
        vision_select_feature = config["vision_select_feature"]
        vision_tower = ViT3DTower(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            vision_select_layer=vision_select_layer,
            vision_select_feature=vision_select_feature,
            pretrained_path=pretrained_path
        )
        return vision_tower
    else:
        raise ValueError(f"Unsupported vision tower config: {config['vision_tower']}")

if __name__=="__main__":
    import torch
    config = {
        "vision_tower": "vit3d",
        "in_channels": 1,
        "img_size": (32, 256, 256),
        "patch_size": (4, 16, 16),
        "hidden_size": 768,
        "num_heads": 8,
        "vision_select_layer": -1,
        "vision_select_feature": "cls_patch"
    }
    weight_path="pretrained_ViT.bin"
    vision_tower = load_vision_encoder(config,pretrained_path=weight_path)
    # print(vision_tower)
    input=torch.randn(2,1,32,256,256)
    output=vision_tower(input)
    print(output.shape)
    
