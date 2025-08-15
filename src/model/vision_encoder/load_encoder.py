from src.model.vision_encoder.vit import ViT3DTower

def load_vision_encoder(config):
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
            vision_select_feature=vision_select_feature
        )
        return vision_tower
    else:
        raise ValueError(f"Unsupported vision tower config: {config['vision_tower']}")