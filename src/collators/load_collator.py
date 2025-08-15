from src.collators.bbox_collator import BboxCollator
from src.collators.white_collator import WhiteCollator
def load_collator(collator_name: str):
    if collator_name == "bbox_2d":
        return BboxCollator()
    elif collator_name=="standard":
        return WhiteCollator()
    else:
        raise ValueError(f"Collator {collator_name} not found")