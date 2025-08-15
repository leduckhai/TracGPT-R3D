import torch
def find_all_linear_names(model):
    """Find all linear layer names for LoRA"""
    cls = torch.nn.Linear
    lora_module_names = set()
    ignore_keywords = [
        "vision_tower",
        "mm_projector",
        "embed_tokens",
        "lm_head",
        "seg_projector",
        "seg_module",
        "bbox3d_head",
        "bbox3d_projector",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)


def print_trainable_params(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"{'Layer name':<60} {'Trainable params':>20} {'% Trainable':>15}")
        print("-" * 100)

        for name, param in model.named_parameters():
            num_params = param.numel()
            if param.requires_grad:
                trainable = "✓"
                percent = num_params / trainable_params * 100
                print(f"{name:<60} {num_params:>20,} {percent:>14.2f}%")
            else:
                if verbose > 1:
                    print(f"{name:<60} {'0':>20} {'(frozen)':>15}")

    print("\nSummary:")
    print(f"Total parameters: {total_params:,}")
    print(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})"
    )
    print(
        f"Frozen parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.1%})"
    )

    return trainable_params, total_params
