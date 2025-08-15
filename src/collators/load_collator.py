from src.collators.bbox_collator import BboxCollator
from src.collators.white_collator import WhiteCollator
from src.collators.standard_collator import StandardCollator
def load_collator(collator_name: str,tokenizer=None):
    if collator_name == "bbox_2d":
        return BboxCollator()
    elif collator_name=="standard":
        return StandardCollator(tokenizer=tokenizer)
    elif collator_name=="white":
        return WhiteCollator()
        
    else:
        raise ValueError(f"Collator {collator_name} not found")