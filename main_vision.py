import os
import argparse
import torch
from transformers import AutoTokenizer
from src.collator import BboxAwareCollator
from torch.utils.data import DataLoader
from src.data.dataloader import load_data
from transformers import TrainingArguments
from eval import evaluate, evaluate_single
import wandb
import numpy as np
from datetime import datetime
from src.model.LanguageModel.Vision_white import TracVisionModel, TracVisionConfig
from src.trainer.vision_trainer import TracVisionTrainer
now = datetime.now()

date_time_string = now.strftime("%d-%m-%Y--%H-%M-%S")
wandb.init(
    project="TracGPT",
    name=f"Trac_llama-{date_time_string}",
)
os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"


def print_info(*args):
    """Simple print function"""
    print(*args)


def create_data_args():
    """Create data arguments namespace"""
    args = argparse.Namespace()
    args.data_root = "./Data/data/"
    args.train_val_dir = "/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/0691cd9f-8dad-4005-811d-34fb610d4f88/train/data"
    args.dataset="trac_white"
    # caption data
    args.cap_data_path = "./Data/data/M3D_Cap_npy/M3D_Cap.json"

    # VQA data
    args.vqa_data_train_path = "./Data/data/M3D-VQA/M3D_VQA_train.csv"
    args.vqa_data_val_path = "./Data/data/M3D-VQA/M3D_VQA_val.csv"
    args.vqa_data_test_path = "./Data/data/M3D-VQA/M3D_VQA_test.csv"
    args.vqa_yn_data_train_path = "./Data/data/M3D-VQA/M3D_VQA_yn_train.csv"

    # positioning & segmentation data
    args.seg_data_path = "./Data/data/M3D_Seg_npy/"
    args.refseg_data_train_path = "./Data/data/M3D_RefSeg_npy/M3D_RefSeg.csv"
    args.refseg_data_test_path = "./Data/data/M3D_RefSeg_npy/M3D_RefSeg_test.csv"

    return args


def set_up_lora(model, training_args):
    print_info("Setting up LoRA...")
    from peft import LoraConfig, get_peft_model, TaskType
    lora_module_names = find_all_linear_names(model)
    # print(f"LoRA target modules: {lora_module_names}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=lora_module_names,
        lora_dropout=0.05,
        bias="lora_only",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"]
    )

    model = get_peft_model(model, lora_config)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}%")

def create_model_args():
    args=argparse.Namespace()
    args.vision_backbone="resnet"    
    args.collator="white"
    
def create_training_args():
    """Create training arguments namespace"""
    args = argparse.Namespace()

    args.lora_enable = True
    args.lora_r = 16
    args.lora_alpha = 32
    args.lora_dropout = 0.05
    args.lora_weight_path = ""
    args.lora_bias = "none"

    args.cache_dir = None
    args.remove_unused_columns = False
    args.model_max_length = 512
    args.seed = 42
    args.optim = "adamw_torch"

    args.bf16 = False
    args.fp16 = True
    args.output_dir = "./output/Tinyllama-finetune-0000/"
    args.num_train_epochs = 3
    args.per_device_train_batch_size = 8
    args.per_device_eval_batch_size = 4
    args.per_device_test_batch_size = 1
    args.gradient_accumulation_steps = 1
    args.evaluation_strategy = "steps"
    args.eval_accumulation_steps = 1
    args.eval_steps = 200
    args.save_strategy = "steps"
    args.save_steps = 1000
    args.save_total_limit = 1
    # args.learning_rate = 5e-5
    args.learning_rate = 1e-4
    args.weight_decay = 0.0
    args.warmup_ratio = 0.03
    args.lr_scheduler_type = "cosine"
    args.logging_steps = 8
    args.gradient_checkpointing = False
    args.dataloader_pin_memory = True
    args.dataloader_num_workers = 8
    args.report_to = "tensorboard"

    args.local_rank = -1
    args.world_size = 1
    args.process_index = 0
    args.n_gpu = 1 if torch.cuda.is_available() else 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.do_train = True
    args.do_eval = True
    args.do_predict = False
    args.overwrite_output_dir = True
    args.load_best_model_at_end = False
    args.metric_for_best_model = None
    args.greater_is_better = None
    args.ignore_data_skip = False
    args.save_safetensors = True
    args.save_on_each_node = False
    args.save_only_model = False
    args.no_cuda = False
    args.use_legacy_prediction_loop = False
    args.prediction_loss_only = False
    args.run_name = None
    args.logging_dir = None
    args.logging_strategy = "steps"
    args.logging_first_step = False
    args.logging_nan_inf_filter = True
    args.include_inputs_for_metrics = False
    args.label_smoothing_factor = 0.0
    args.debug = []
    args.sharded_ddp = []
    args.fsdp = []
    args.fsdp_config = {}
    args.deepspeed = None
    args.label_names = None
    args.resume_from_checkpoint = None
    args.hub_model_id = None
    args.hub_strategy = "every_save"
    args.hub_token = None
    args.hub_private_repo = False
    args.hub_always_push = False
    args.gradient_checkpointing_kwargs = None
    args.include_num_input_tokens_seen = False
    args.neftune_noise_alpha = None
    args.optim_args = None
    args.ray_scope = "last"
    args.ddp_timeout = 1800
    args.torch_compile = False
    args.torch_compile_backend = None
    args.torch_compile_mode = None
    args.dispatch_batches = None
    args.split_batches = False
    args.include_tokens_per_second = False
    args.should_save = True
    return args


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
                if verbose > 1:  # Only show frozen layers if verbose > 1
                    print(f"{name:<60} {'0':>20} {'(frozen)':>15}")
    
    print("\nSummary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})")
    print(f"Frozen parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params:.1%})")
    
    return trainable_params, total_params

def parse_arguments():
    """Enhanced argument parser with all new parameters"""
    parser = argparse.ArgumentParser(
        description="Medical LLM Training with Enhanced Parameters"
    )

    parser.add_argument("--version", type=str, default="v0", help="Model version")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/phi-2",
        help="Model name or path",
    )
    parser.add_argument("--model_type", type=str, default="phi3", help="Model type")
    parser.add_argument(
        "--vision_tower", type=str, default="vit3d", help="Vision tower type"
    )
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone"
    )
    parser.add_argument(
        "--tune_mm_mlp_adapter", action="store_true", help="Tune MM MLP adapter"
    )
    parser.add_argument(
        "--model_max_length", type=int, default=512, help="Maximum model length"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_enable",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable LoRA",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Data arguments
    parser.add_argument(
        "--data_root", type=str, default="./Data/data/", help="Data root directory"
    )
    parser.add_argument(
        "--cap_data_path",
        type=str,
        default="./Data/data/M3D_Cap_npy/M3D_Cap.json",
        help="Caption data path",
    )

    # Training arguments
    parser.add_argument("--bf16", type=int, default=0, help="Use bf16 (0 or 1)")
    parser.add_argument("--fp16", type=int, default=1, help="Use fp16 (0 or 1)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/tinyllama-0000",
        help="Output directory",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Train batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Eval batch size per device",
    )

    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=2,
        help="test batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--evaluation_strategy", type=str, default="steps", help="Evaluation strategy"
    )
    parser.add_argument(
        "--eval_accumulation_steps", type=int, default=1, help="Eval accumulation steps"
    )
    parser.add_argument("--eval_steps", type=float, default=4, help="Eval steps")
    parser.add_argument(
        "--save_strategy", type=str, default="steps", help="Save strategy"
    )
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument(
        "--save_total_limit", type=int, default=1, help="Save total limit"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type"
    )
    parser.add_argument("--logging_steps", type=float, default=4, help="Logging steps")
    parser.add_argument(
        "--gradient_checkpointing",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Gradient checkpointing",
    )
    # what is dataloader_pin_memory?

    parser.add_argument(
        "--dataloader_pin_memory",
        type=lambda x: x.lower() == "False",
        default=True,
        help="Pin memory",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--report_to", type=str, default="wandb", help="Reporting platform"
    )

    return parser.parse_args()

    

def main():
    
    cmd_args = parse_arguments()
    model_args=create_model_args()
    data_args = create_data_args()
    training_args = create_training_args()

    print("training_args:", training_args)

    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_args.seed)
    print("MODEL ARGS:",model_args)
    
    print_info("=" * 20 + " Enhanced Training Setup " + "=" * 20)
    print_info(f"Device: {training_args.device}")
    print_info(f"Base Model{cmd_args.model_name_or_path} ")
    print_info(f"LoRA Enabled: {training_args.lora_enable}")
    print_info(f"BF16: {training_args.bf16}, FP16: {training_args.fp16}")
    print_info(f"Output: {training_args.output_dir}")
    print_info(f"Epochs: {training_args.num_train_epochs}")
    print_info(f"Train Batch Size: {training_args.per_device_train_batch_size}")
    print_info(f"Eval Batch Size: {training_args.per_device_eval_batch_size}")
    print_info(f"Learning Rate: {training_args.learning_rate}")
    print_info(f"Save Steps: {training_args.save_steps}")
    print_info(f"Dataloader Workers: {training_args.dataloader_num_workers}")

    print_info("=" * 20 + " Tokenizer preparation " + "=" * 20)
    tokenizer = AutoTokenizer.from_pretrained(
        cmd_args.model_name_or_path,

    )

    special_tokens = [
        "<im_patch>",
        "<end>",
    ]
    image_token_name = "<im_patch>"
    num_added = tokenizer.add_tokens(special_tokens)

    print(f"Added {num_added} special tokens", len(tokenizer))
    if model_args.collator=="bbox":
        
        collator = BboxAwareCollator(
            tokenizer=tokenizer,
            max_length=cmd_args.model_max_length,
            max_bbox_length=9,
            num_vision_token=256,
            token_name=image_token_name,
            one_bbox=True
        )
    elif model_args.collator=="white":
        from collator import WhiteCollator
        collator=WhiteCollator()
    else:
        raise NotImplementedError
    
    train_set, val_set, test_set = load_data(bbox_only=True)
    print("train set", len(train_set))
    print("val set", len(val_set))
    print("test set", len(test_set))
 
    test_loader = DataLoader(
        test_set,
        batch_size=training_args.per_device_test_batch_size,
        collate_fn=collator,
        pin_memory=cmd_args.dataloader_pin_memory,
    )

    # img_token_id = tokenizer.convert_tokens_to_ids(image_token_name)
    if model_args.vision_backbone=="resnet":
        from src.model.Encoder.resnet import ResNet18_3D
        model=ResNet18_3D()
    elif model_args.vision_backbone=="densenet":
        from src.model.Encoder.densenet import DenseNet3D
    else:
        raise NotImplementedError
    # model=TracVisionModel()
    print("trainable params", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Layer grad",print_trainable_params(model))
    wandb.watch(
        model,
        log="all",       # Logs gradients + parameters
        log_freq=10,     # Log every 10 steps
        log_graph=True,  # Optional: Log computation graph
    )

    if cmd_args.freeze_backbone:
        print_info("Freezing backbone...")
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # if training_args.lora_enable:
    #     set_up_lora(model, training_args)

    model.to(training_args.device)
               
    trainer = TracVisionTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=training_args.output_dir,
            # max_grad_norm=4.0, 
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            num_train_epochs=training_args.num_train_epochs,
            logging_dir=os.path.join(training_args.output_dir, "logs"),
            # evaluation_strategy=training_args.evaluation_strategy, 
            eval_steps=(
                int(training_args.eval_steps) if training_args.eval_steps else None
            ),
            logging_steps=int(training_args.logging_steps),
            save_strategy=training_args.save_strategy,
            save_steps=training_args.save_steps,
            # fp16=training_args.fp16,
            # bf16=training_args.bf16,
            fp16=False,  # Disable if currently True
            bf16=False,  # Disable if currently True
            learning_rate=training_args.learning_rate,  # Added learning_rate
            weight_decay=training_args.weight_decay,  # Added weight_decay
            warmup_ratio=training_args.warmup_ratio,  # Added warmup_ratio
            lr_scheduler_type=training_args.lr_scheduler_type,  # Added lr_scheduler_type
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,  # Added gradient_accumulation_steps
            gradient_checkpointing=training_args.gradient_checkpointing,  # Added gradient_checkpointing
            dataloader_num_workers=training_args.dataloader_num_workers,  # Added dataloader_num_workers
            save_total_limit=training_args.save_total_limit,  # Added save_total_limit
            load_best_model_at_end=training_args.load_best_model_at_end,
            report_to=["wandb"],
            remove_unused_columns=training_args.remove_unused_columns,  # Added remove_unused_columns
            seed=training_args.seed,  
            save_safetensors=False,
            dataloader_pin_memory=False,
        ),
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    # torch.autograd.set_detect_anomaly(True, check_nan=True)
    trainer.train()
    print_info("Training complete!")
    print("evaluate")
    metrics = trainer.evaluate()
    print(f"Loss: {metrics['eval_loss']}")
    print(f"IoU: {metrics['eval_iou']}")

    evaluate(
        model=model,
        data_loader=test_loader,
        tokenizer=tokenizer,
        save_path="generate_output",
    )
    # Save the final model
    print_info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()