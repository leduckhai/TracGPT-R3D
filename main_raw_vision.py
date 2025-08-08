import os
import argparse
import torch
from torch.utils.data import DataLoader
from src.data.dataloader import load_data
from transformers import TrainingArguments

# from eval import evaluate, evaluate_single
from datetime import datetime
from src.trainers.raw_vision_trainer import RawVisionTrainer
from src.model.Encoder.resnet import ResNet18_3D
from src.collator import WhiteCollator
from src.trainers.tracker import WandbTracker
import argparse
from dataclasses import dataclass
import yaml
import json
from pathlib import Path
from transformers import TrainingArguments
# import datetime 
from datetime import datetime

os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"


def print_info(*args):
    """Simple print function"""
    print(*args)


@dataclass
class DataConfig:
    data_root: str = "./data/"
    dataset: str = "trac_white"
    train_val_split: float = 0.8
    train_val_dir: str = ""
    train_sample: int = -1
    val_sample: int = -1
    test_sample: int = -1
    dataloader_pin_memory: bool = True


@dataclass
class ModelConfig:
    vision_backbone: str = "resnet50"
    collator: str = "white"
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    tags: list=[]


# ======================
# 2. CLI Argument Parsing
# ======================
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument(
        "--config_path", type=str, default="/root/TracGPT-R3D/config/white.yaml"
    )

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--dataset", type=str)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)

    parser.add_argument("--lora_enable", action="store_true")
    parser.add_argument("--lora_r", type=int)

    return parser.parse_args()


# ======================
# 3. Config Initialization
# ======================
def load_configs():
    now = datetime.now()

    # Format as string (YYYY-MM-DD_HH-MM-SS)
    datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    cli_args = parse_cli_args()
    config_path = cli_args.config_path
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    data_config = DataConfig(**yaml_config["data"])

    model_config = ModelConfig(**yaml_config["model"])
    train_config = yaml_config["training"]
    train_config["logging_dir"] = os.path.join(train_config["output_dir"], "logs")
    train_config["learning_rate"]=float(train_config["learning_rate"])
    train_config["output_dir"]=os.path.join(train_config["output_dir"],datetime_str)
    print("OUTPUT DIR", train_config["output_dir"])
    training_config = TrainingArguments(**train_config)

    return training_config, model_config, data_config

def save_configs(output_dir: str, training_config, model_config, data_config, cli_args):
    config = {
        "cli_args": vars(cli_args),
        "model_config": vars(model_config),
        "data_config": vars(data_config),
        "training_config": training_config.to_dict(),
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)


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


def main():

    training_config, model_config, data_config = load_configs()
    cli_args = parse_cli_args()
    print("TRAIN CONFIG", training_config)
    print("MODEL CONFIG", model_config)
    print("DATA CONFIG", data_config)
    save_configs(
        training_config.output_dir, training_config, model_config, data_config, cli_args
    )

    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config.seed)

    print_info("=" * 20 + " Tokenizer preparation " + "=" * 20)
    if model_config.collator == "white":
        collator = WhiteCollator()
    else:
        raise NotImplementedError
    # train_set, val_set, test_set = load_data(train_val_dir=data_config.train_val_dir,dataset=data_config.dataset, train_sample=5,val_sample=5)
    train_set, val_set, test_set = load_data(
        train_val_dir=data_config.train_val_dir,
        dataset=data_config.dataset,
        train_sample=data_config.train_sample,
        val_sample=data_config.val_sample,
        test_sample=data_config.test_sample,
    )
    print("OUTPUT DIR",training_config.output_dir)
    print("train set", len(train_set))
    print("val set", len(val_set))
    print("test set", len(test_set))

    test_loader = DataLoader(
        test_set,
        batch_size=training_config.per_device_eval_batch_size,
        collate_fn=collator,
        pin_memory=data_config.dataloader_pin_memory,
    )
    if training_config.report_to[0] == "wandb":
        tracker = WandbTracker(tags=model_config.tags)
    else:
        raise NotImplementedError(f"Tracker is not match{training_config.report_to}")

    print_info("=" * 20 + " Model preparation " + "=" * 20)
    if model_config.vision_backbone == "resnet":
        model = ResNet18_3D()
    # elif model_config.vision_backbone=="densenet":
    #     from src.model.Encoder.densenet import DenseNet3D
    else:
        raise NotImplementedError(model_config.vision_backbone)
    print(
        "trainable params",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("Layer grad", print_trainable_params(model))

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(training_config.device)

    batch_size = training_config.per_device_train_batch_size * max(
        1, training_config.n_gpu
    )
    train_dataset_size = len(train_set)
    gradient_accumulation_steps = training_config.gradient_accumulation_steps or 1

    steps_per_epoch = train_dataset_size // (batch_size * gradient_accumulation_steps)

    eval_steps = max(1, steps_per_epoch // 4)
    print("EVAL STEP",eval_steps)
    training_config.eval_steps = eval_steps
    training_config.save_steps = eval_steps
    trainer = RawVisionTrainer(
        model=model,
        tracker=tracker,
        args=training_config,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collator,
    )
    # torch.autograd.set_detect_anomaly(True, check_nan=True)
    trainer.train()
    print_info("Training complete!")
    print("evaluate")
    metrics = trainer.evaluate()
    tracker.on_train_end()
    # evaluate(
    #     model=model,
    #     data_loader=test_loader,
    #     tokenizer=tokenizer,
    #     save_path="generate_output",
    # )
    # Save the final model
    print_info(f"Model saved to {training_config.output_dir}")


if __name__ == "__main__":
    main()
