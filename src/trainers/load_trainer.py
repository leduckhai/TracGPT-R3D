from src.trainers.bbox_2d_trainer import Bbox2DTrainer
def load_trainer(trainer_name: str):
    # if trainer_name == "vlmtrac":
    #     from .vlmtrac import VLMTracTrainer
    #     return VLMTracTrainer
    # else:
    
    if trainer_name == "bbox_2d":
        return Bbox2DTrainer
    else:
        raise ValueError(f"Trainer {trainer_name} not found")