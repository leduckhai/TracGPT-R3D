from src.trainers.bbox_2d_trainer import Bbox2DTrainer
from src.trainers.standard_trainer import StandardTrainer
def load_trainer(trainer_name: str):
    # if trainer_name == "vlmtrac":
    #     from .vlmtrac import VLMTracTrainer
    #     return VLMTracTrainer
    # else:
    
    if trainer_name == "bbox_2d":
        return Bbox2DTrainer
    elif trainer_name == "standard":
        return StandardTrainer
    else:
        raise ValueError(f"Trainer {trainer_name} not found")