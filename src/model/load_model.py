from src.model.vision_encoder.rcnn import TrainableFasterRCNN
def load_model(config):
    if config.vision_backbone == "rcnn":
        model = TrainableFasterRCNN()
        return model
    else:
        raise ValueError(f"Model {config.vision_backbone} not found")
    