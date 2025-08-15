import os
import argparse
import torch
from src.dataset.dataloader import load_data
from transformers import TrainingArguments
from datetime import datetime
from src.trainers.raw_vision_trainer import RawVisionTrainer
from src.trainers.tracker import WandbTracker
import argparse
from dataclasses import dataclass
import yaml
import json
from pathlib import Path
from transformers import TrainingArguments
from datetime import datetime
from dataclasses import dataclass, field
from typing import List
from src.utils.printer import print_trainable_params, find_all_linear_names
from src.collators.load_collator import load_collator
from src.trainers.load_trainer import load_trainer
from src.model.load_model import load_model

os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"


def print_info(*args):
    """Simple print function"""
    print(*args)


@dataclass
class GeneralConfig:
    max_eval: int = 4
    trainer: str = "raw_vision"


@dataclass
class DataConfig:
    data_root: str = "./data/"
    dataset: str = "trac_white"
    dataset_config: dict = field(default_factory=dict)
    train_val_split: float = 0.8
    train_val_dir: str = ""
    test_dir: str = ""
    image_train_path: str = ""
    image_test_path: str = ""
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
    tags: List[str] = field(default_factory=list)


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


def load_configs():
    now = datetime.now()

    datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    cli_args = parse_cli_args()
    config_path = cli_args.config_path
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    data_config = DataConfig(**yaml_config["data"])

    model_config = ModelConfig(**yaml_config["model"])
    general_config = GeneralConfig(**yaml_config["general"])

    train_config = yaml_config["training"]
    train_config["logging_dir"] = os.path.join(train_config["output_dir"], "logs")
    train_config["learning_rate"] = float(train_config["learning_rate"])
    # train_config["output_dir"] = os.path.join(train_config["output_dir"], datetime_str)
    # print("OUTPUT DIR", train_config["output_dir"])
    training_config = TrainingArguments(**train_config)

    return training_config, model_config, data_config, general_config


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


def main():

    training_config, model_config, data_config, general_config = load_configs()
    cli_args = parse_cli_args()
    print("TRAIN CONFIG", training_config)
    print("MODEL CONFIG", model_config)
    print("DATA CONFIG", data_config)
    print("GENERAL CONFIG", general_config)
    save_configs(
        training_config.output_dir, training_config, model_config, data_config, cli_args
    )

    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config.seed)
    train_set, val_set, test_set = load_data(
        train_val_dir=data_config.train_val_dir,
        test_dir=data_config.test_dir,
        image_train_path=data_config.image_train_path,
        image_test_path=data_config.image_test_path,
        dataset=data_config.dataset,
        train_sample=data_config.train_sample,
        val_sample=data_config.val_sample,
        test_sample=data_config.test_sample,
        dataset_config=data_config.dataset_config,
    )
    print("OUTPUT DIR", training_config.output_dir)
    print("train set", len(train_set))
    print("val set", len(val_set))
    print("test set", len(test_set))

    run_id=None
    if training_config.report_to[0] == "wandb":
        print("tags", model_config.tags)
        tracker = WandbTracker(tags=model_config.tags)
        run_id = tracker.get_id()
        print_info(f"Wandb run ID: {run_id}")
        output_dir=os.path.join(training_config.output_dir, run_id)
        training_config.output_dir = output_dir
        print("Updated output directory:", training_config.output_dir)
    else:
        raise NotImplementedError(f"Tracker is not match{training_config.report_to}")

    print_info("=" * 20 + " Model preparation " + "=" * 20)
    model = load_model(
        config=model_config,
    )
    collator = load_collator(
        collator_name=model_config.collator,
    )
    custom_trainer = load_trainer(trainer_name=general_config.trainer)
    print(
        "trainable params",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("Layer grad", print_trainable_params(model))

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(training_config.device)

    if general_config.max_eval != -1:
        batch_size = training_config.per_device_train_batch_size * max(
            1, training_config.n_gpu
        )
        train_dataset_size = len(train_set)
        gradient_accumulation_steps = training_config.gradient_accumulation_steps or 1

        steps_per_epoch = train_dataset_size // (
            batch_size * gradient_accumulation_steps
        )

        eval_steps = max(5, steps_per_epoch // general_config.max_eval)

        print("EVAL STEP", eval_steps)
        training_config.eval_steps = eval_steps
        training_config.save_steps = eval_steps
    trainer = custom_trainer(
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
    # test_results = trainer.evaluate(eval_dataset=test_set, metric_key_prefix="test")
    val_results = trainer.evaluate()
    
    tracker.on_train_end()

    print_info(f"Model saved to {training_config.output_dir}")


if __name__ == "__main__":
    main()
