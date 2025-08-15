from src.collators.bbox_collator import BboxCollator
def load_collator(collator_name: str):
    if collator_name == "bbox_2d":
        return BboxCollator()
    else:
        raise ValueError(f"Collator {collator_name} not found")