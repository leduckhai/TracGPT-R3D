from src.collators.standard_collator import StandardCollator
def load_collator(collator_name: str,tokenizer=None):
    if collator_name=="standard":
        return StandardCollator(tokenizer=tokenizer)
    else:
        raise ValueError(f"Collator {collator_name} not found")