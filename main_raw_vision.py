import os
import argparse
import torch
from src.dataset.dataloader import load_data
from transformers import TrainingArguments
from datetime import datetime
from src.trainers.tracker import WandbTracker, DummyTracker
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
from types import SimpleNamespace
from dataclasses import dataclass, asdict
# from inference import infer_test_data
from transformers import EarlyStoppingCallback

os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def print_info(*args):
    """Simple print function"""
    print(*args)


@dataclass
class GeneralConfig:
    max_eval: int = 4
    trainer: str = "standard"
    collator: str = "standard"
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_bias: str = "none"


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
    name: str = "vit_llama"
    config: dict = field(default_factory=dict)
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    tags: List[str] = field(default_factory=list)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument(
        "--config_path", type=str, default="/root/TracGPT-R3D/config/white.yaml"
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--use_lora", type=bool, default=True, help="Use LoRA in pretrained model")  
    return parser.parse_args()


def load_configs():
    now = datetime.now()
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
    training_config = TrainingArguments(**train_config)
    return training_config, model_config, data_config, general_config

def set_up_lora(model, lora_r, lora_alpha, lora_dropout, lora_target_modules, lora_bias):
    print_info("Setting up LoRA...")
    from peft import LoraConfig, get_peft_model, TaskType
    print("lora r", lora_r, "lora alpha", lora_alpha, "lora dropout", lora_dropout, "lora target modules", lora_target_modules, "lora bias", lora_bias)
    # lora_module_names = find_all_linear_names(model)
    # print(f"LoRA target modules: {lora_module_names}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha= lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias= lora_bias,
        init_lora_weights=True,
        task_type=TaskType.CAUSAL_LM,
        # modules_to_save=["embed_tokens", "lm_head"]
    )
    model.lm_model = get_peft_model(model.lm_model, lora_config)
    # trainable_params, all_params = model.lm_model.get_nb_trainable_parameters()
    # print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}%")
    return model

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


# def find_largest_batch_size(model_init, args, dataset, tokenizer, max_batch_size=128):
#     batch_size = 1
#     best_batch_size = batch_size
#     oom = False

#     while batch_size <= max_batch_size and not oom:
#         try:
#             print(f"Trying batch size = {batch_size}")
#             args.per_device_train_batch_size = batch_size
#             trainer = Trainer(
#                 model_init=model_init,
#                 args=args,
#                 train_dataset=dataset,
#                 tokenizer=tokenizer,
#             )
#             trainer.train(max_steps=1)  # Just a quick test step
#             best_batch_size = batch_size
#             batch_size *= 2
#         except torch.cuda.OutOfMemoryError:
#             print(f"OOM at batch size = {batch_size}")
#             torch.cuda.empty_cache()
#             oom = True

#     print(f"✅ Best batch size: {best_batch_size}")
#     return best_batch_size

def main():

    training_config, model_config, data_config, general_config = load_configs()
    cli_args = parse_cli_args()
    is_test = cli_args.test
    pretrained_path = cli_args.pretrained_path
    use_lora = cli_args.use_lora

    if is_test:
        print("Running in test mode")
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
    if not is_test:
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
        tokenizer,model = load_model(
            config=asdict(model_config))
        collator = load_collator(
            collator_name=general_config.collator,
            tokenizer=tokenizer,
        )
        if general_config.lora:
            model = set_up_lora(
                model=model,
                lora_r=general_config.lora_r,
                lora_alpha=general_config.lora_alpha,
                lora_dropout=general_config.lora_dropout,
                lora_target_modules=general_config.lora_target_modules,
                lora_bias=general_config.lora_bias
            )
       
        model.to("cuda")
        
        # if general_config.max_eval != -1:
        #     batch_size = training_config.per_device_train_batch_size * max(
        #         1, training_config.n_gpu
        #     )
        #     train_dataset_size = len(train_set)
        #     gradient_accumulation_steps = training_config.gradient_accumulation_steps or 1

        #     steps_per_epoch = train_dataset_size // (
        #         batch_size * gradient_accumulation_steps
        #     )

        #     eval_steps = max(1, steps_per_epoch // general_config.max_eval)

        #     print("EVAL STEP", eval_steps)
        #     training_config.eval_steps = eval_steps
        #     training_config.save_steps = eval_steps
            
        # model.freeze_llm()
        custom_trainer = load_trainer(trainer_name=general_config.trainer)
        
        trainer = custom_trainer(
            model=model,
            tracker=tracker,
            args=training_config,
            tokenizer=tokenizer,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        print_trainable_params(model)
        trainer.train()
        trainer.inference(eval_dataset=test_set )
        tracker.on_train_end()
        print_info(f"Model saved to {training_config.output_dir}")

    else:
        print_info("=" * 20 + " Model preparation " + "=" * 20)
        tracker= DummyTracker()
        id = tracker.get_id()
        print_info(f"Dummy Tracker ID: {id}")
        output_dir = training_config.output_dir
        output_path = os.path.join(output_dir, id)
        os.makedirs(output_path, exist_ok=True)
        tokenizer, model = load_model(
            config=asdict(model_config),
            pretrained_path=pretrained_path,
            lora=use_lora
        )
        collator = load_collator(
            collator_name=general_config.collator,
            tokenizer=tokenizer,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=training_config.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=data_config.dataloader_pin_memory,
        )
        print_info("Test dataloader created")
        
        # test_results = infer_test_data(model=model,tokenizer=tokenizer, dataloader=test_dataloader, output_dir=output_path)
 

if __name__ == "__main__":
    main()
