import os
import logging
import argparse
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
import json
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import Phi3Config, Phi3Model, Phi3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from model.LanguageModel.Trac_phi3 import TracPhi3ForCausalLM, TracPhi3Config
from collator import QA3DDataset, BboxAwareCollator
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Union, Tuple

# Disable distributed training detection
os.environ["RANK"] = "-1"
os.environ["LOCAL_RANK"] = "-1"
os.environ["WORLD_SIZE"] = "1"


def print_info(*args):
    """Simple print function"""
    print(*args)


def create_model_args():
    """Create model arguments namespace"""
    args = argparse.Namespace()
    args.version = "v0"
    args.model_name_or_path = "microsoft/phi-2"
    args.model_type = "phi3"  # Changed from phi2 to phi3 to match architecture
    args.freeze_backbone = False
    args.pretrain_mllm = None
    args.tune_mm_mlp_adapter = False
    args.pretrain_mm_mlp_adapter = None

    # image
    args.image_channel = 1
    args.image_size = (32, 256, 256)
    args.patch_size = (4, 16, 16)

    # vision
    args.vision_tower = "vit3d"
    args.vision_select_layer = -1
    args.vision_select_feature = "patch"
    args.pretrain_vision_model = None
    args.freeze_vision_tower = False
    args.num_new_tokens = 4
    args.hidden_size=768  
    args.num_heads = 12  # Number of attention heads, can be adjusted based on model size

    # projector
    args.mm_projector_type = "spp"
    args.proj_layer_type = "mlp"
    args.proj_layer_num = 2
    args.proj_pooling_type = "spatial"
    args.proj_pooling_size = 2

    # segvol
    args.segmentation_module = None
    args.pretrain_seg_module = None

    # bbox3d
    args.bbox3d_module = "simple"  # Enable bbox3d module
    args.hidden_size = 1024
    args.num_classes = 1
    args.max_bbox_length = 9
    args.mm_hidden_size = 2560

    return args


def create_data_args():
    """Create data arguments namespace"""
    args = argparse.Namespace()
    args.data_root = "./Data/data/"

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


def create_training_args():
    """Create training arguments namespace"""
    args = argparse.Namespace()

    # lora
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

    # Training configuration - updated with new parameters
    args.bf16 = False  # Changed to False (0)
    args.fp16 = True  # Changed to True (1)
    args.output_dir = "./LaMed/output/LaMed-Phi3-4B-finetune-0000"
    args.num_train_epochs = 5
    args.per_device_train_batch_size = 8
    args.per_device_eval_batch_size = 4
    args.gradient_accumulation_steps = 1
    args.evaluation_strategy = "steps"
    args.eval_accumulation_steps = 1
    args.eval_steps = 0.04
    args.save_strategy = "steps"
    args.save_steps = 1000
    args.save_total_limit = 1
    args.learning_rate = 5e-5
    args.weight_decay = 0.0
    args.warmup_ratio = 0.03
    args.lr_scheduler_type = "cosine"
    args.logging_steps = 0.001
    args.gradient_checkpointing = False
    args.dataloader_pin_memory = True
    args.dataloader_num_workers = 8
    args.report_to = "tensorboard"

    # Single process training
    args.local_rank = -1
    args.world_size = 1
    args.process_index = 0
    args.n_gpu = 1 if torch.cuda.is_available() else 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Additional required attributes for Trainer
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


def create_trac_phi3_config(model_args):
    """Create TracPhi3Config from model arguments"""
    # Start with base Phi3 config
    base_config = {
        'hidden_size': model_args.hidden_size,
        'intermediate_size': model_args.hidden_size * 4,
        'num_attention_heads': 32,
        'num_hidden_layers': 32,
        'num_key_value_heads': 32,
        'vocab_size': model_args.vocab_size,
        'max_position_embeddings': 4096,
        'rms_norm_eps': 1e-5,
        'rope_theta': 10000.0,
        'sliding_window': None,
        'attention_dropout': 0.0,
        'return_dict': True,
        'output_hidden_states': False,
        'output_attentions': False,
        'torch_dtype': 'float16',
        'use_cache': True,
    }
    
    # Add multimodal configuration
    multimodal_config = {
        'vision_tower': model_args.vision_tower,
        'mm_projector_type': model_args.mm_projector_type,
        'bbox3d_module': model_args.bbox3d_module,
        'mm_hidden_size': model_args.mm_hidden_size,
        'bbox3d_token_id': getattr(model_args, 'bbox3d_token_id', None),
        'img_token_id': model_args.img_token_id,
        'image_channel': model_args.image_channel,
        'image_size': model_args.image_size if isinstance(model_args.image_size, int) else model_args.image_size[1],
        'patch_size': model_args.patch_size if isinstance(model_args.patch_size, int) else model_args.patch_size[1],
        'vision_select_layer': model_args.vision_select_layer,
        'vision_select_feature': model_args.vision_select_feature,
        'proj_layer_type': model_args.proj_layer_type,
        'proj_layer_num': model_args.proj_layer_num,
        'proj_pooling_type': model_args.proj_pooling_type,
        'proj_pooling_size': model_args.proj_pooling_size,

        # vision config 
        'hidden_size':model_args.hidden_size,
        'nums_heads': model_args.num_heads,
    }
    
    # Combine all configurations
    config_dict = {**base_config, **multimodal_config}
    return TracPhi3Config(**config_dict)


def compute_metrics(eval_preds):
    """Compute accuracy metrics"""
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds == filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}


def preprocess_logits_for_metrics(logits, labels):
    """Preprocess logits for metrics computation"""
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    """Handle DeepSpeed zero optimization"""
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if not ignore_status:
                    logging.warning(
                        f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                    )
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
    except ImportError:
        param = param.detach().cpu().clone()
    return param


def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    """Get projector state with optional DeepSpeed handling"""
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    """Save model safely"""
    os.makedirs(output_dir, exist_ok=True)

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ["mm_projector", "embed_tokens"]
        weight_to_save = get_mm_projector_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)

        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    # Standard model saving
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    del state_dict

    # Save model and tokenizer
    trainer.model.save_pretrained(output_dir, state_dict=cpu_state_dict)
    if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(output_dir)


def find_all_linear_names(model):
    """Find all linear layer names for LoRA"""
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
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


def parse_arguments():
    """Enhanced argument parser with all new parameters"""
    parser = argparse.ArgumentParser(
        description="Medical LLM Training with Enhanced Parameters"
    )

    # Model arguments
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
        default="./LaMed/output/LaMed-Phi3-4B-finetune-0000",
        help="Output directory",
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=5.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Train batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Eval batch size per device",
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
    parser.add_argument("--eval_steps", type=float, default=0.04, help="Eval steps")
    parser.add_argument(
        "--save_strategy", type=str, default="steps", help="Save strategy"
    )
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument(
        "--save_total_limit", type=int, default=1, help="Save total limit"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type"
    )
    parser.add_argument(
        "--logging_steps", type=float, default=0.001, help="Logging steps"
    )
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
        "--report_to", type=str, default="tensorboard", help="Reporting platform"
    )
    parser.add_argument(
        "--model_max_length", type=int, default=512, help="Max model length"
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    cmd_args = parse_arguments()

    # Create argument namespaces
    model_args = create_model_args()
    data_args = create_data_args()
    training_args = create_training_args()
    print("model_args:", model_args)
    print("data_args:", data_args)
    print("training_args:", training_args)

    # Override with command line arguments
    for key, value in vars(cmd_args).items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)
        elif hasattr(data_args, key):
            setattr(data_args, key, value)
        elif hasattr(training_args, key):
            # Handle special conversions for training args
            if key == "bf16":
                setattr(training_args, key, bool(value))
            elif key == "fp16":
                setattr(training_args, key, bool(value))
            else:
                setattr(training_args, key, value)

    # Set random seed
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_args.seed)

    print_info("=" * 20 + " Enhanced Training Setup " + "=" * 20)
    print_info(f"Device: {training_args.device}")
    print_info(f"Model: {model_args.model_name_or_path}")
    print_info(f"Model Type: {model_args.model_type}")
    print_info(f"Version: {model_args.version}")
    print_info(f"Vision Tower: {model_args.vision_tower}")
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
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Add special tokens
    special_token = {
        "additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>","<image>", "<image_newline>"]
    }
    tokenizer.add_special_tokens(special_token)
    tokenizer.add_tokens("[SEG]")
    # image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_newline_token_id = tokenizer.convert_tokens_to_ids("<image_newline>")


    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if model_args.model_type and "llama3" in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Set token IDs
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.bbox3d_token_id = tokenizer.convert_tokens_to_ids("<bx_start>")
    model_args.vocab_size = len(tokenizer)
    
    print_info("img_token_id:", model_args.img_token_id)
    print_info("seg_token_id:", model_args.seg_token_id) 
    print_info("bbox3d_token_id:", model_args.bbox3d_token_id)
    print_info("vocab_size:", model_args.vocab_size)

    print_info("=" * 20 + " Model preparation " + "=" * 20)

    # Create TracPhi3Config
    config = create_trac_phi3_config(model_args)
    
    # Set bbox3d_token_id in config
    config.bbox3d_token_id = model_args.bbox3d_token_id

    # Load model with custom config
    print_info("Loading TracPhi3ForCausalLM model...")
    model = TracPhi3ForCausalLM(config)
    
    # Load pretrained weights from the base model
    if model_args.model_name_or_path and model_args.model_name_or_path != "":
        try:
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=(
                    torch.bfloat16
                    if training_args.bf16
                    else (torch.float16 if training_args.fp16 else torch.float32)
                ),
            )
            
            # Load compatible weights
            model_dict = model.state_dict()
            pretrained_dict = base_model.state_dict()
            
            # Filter out incompatible keys and load compatible ones
            compatible_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                    
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict, strict=False)
            print_info(f"Loaded {len(compatible_dict)} compatible weights from pretrained model")
            
            del base_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print_info(f"Warning: Could not load pretrained weights: {e}")

    # Resize token embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Set model configuration
    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Initialize multimodal components
    print_info("Initializing multimodal components...")
    model.get_model().initialize_multimodal_components(model_args)
    print_info("Multimodal components initialized successfully!")

    # LoRA setup
    if training_args.lora_enable:
        print_info("Setting up LoRA...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # Find all linear layer names for LoRA
            lora_module_names = find_all_linear_names(model)
            print_info(f"LoRA target modules: {lora_module_names}")

            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=lora_module_names,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, lora_config)
            print_info("LoRA setup complete!")
            print_info(f"Trainable parameters: {model.print_trainable_parameters()}")

        except ImportError:
            print_info("Warning: PEFT not available, LoRA disabled")
            training_args.lora_enable = False

    # Move model to device
    model = model.to(training_args.device)
    
    # Setup data collator and dataset
    print_info("=" * 20 + " Data preparation " + "=" * 20)
    collator = BboxAwareCollator(
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        max_bbox_length=9,
    )
    
    try:
        ds = QA3DDataset()
        dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collator)
        print_info("Dataset and DataLoader created successfully!")
    except Exception as e:
        print_info(f"Warning: Failed to create dataset: {e}")
        print_info("Creating dummy dataloader for testing...")
        
        # Create a simple dummy dataset for testing
        class DummyDataset:
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return {
                    'images': torch.randn(1, 32, 256, 256),
                    'input_ids': torch.randint(0, 1000, (20,)),
                    'attention_masks': torch.ones(20),
                    'labels': torch.randint(0, 1000, (20,)),
                    'bbox_gts': torch.randn(1, 6),
                    'bbox_masks': torch.ones(1).bool(),
                    'answer_types': ['bbox'],
                    'position_ids': torch.arange(20)
                }
        
        dummy_ds = DummyDataset()
        
        def dummy_collate_fn(batch):
            # Simple collation for testing with proper tensor cloning
            return {
                'images': torch.stack([item['images'].clone() for item in batch]),           # Added .clone()
                'input_ids': torch.stack([item['input_ids'].clone() for item in batch]),     # Added .clone()
                'attention_mask': torch.stack([item['attention_masks'].clone() for item in batch]),  # Fixed key name + .clone()
                'labels': torch.stack([item['labels'].clone() for item in batch]),           # Added .clone()
                'bbox_gts': torch.stack([item['bbox_gts'].clone() for item in batch]),       # Added .clone()
                'bbox_masks': torch.stack([item['bbox_masks'].clone() for item in batch]),   # Added .clone()
                'answer_types': [item['answer_types'][0] for item in batch],
                'position_ids': torch.stack([item['position_ids'].clone() for item in batch]),  # Added .clone(), removed duplicate
            }

        dl = DataLoader(dummy_ds, batch_size=2, shuffle=True, collate_fn=dummy_collate_fn)

    # Define a basic Trainer setup
    print_info("=" * 20 + " Trainer Setup " + "=" * 20)
    from transformers import Trainer, TrainingArguments

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=training_args.output_dir,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            num_train_epochs=training_args.num_train_epochs,
            logging_dir=os.path.join(training_args.output_dir, "logs"),
            eval_strategy=training_args.evaluation_strategy,  # Changed from evaluation_strategy
            eval_steps=max(1, int(training_args.eval_steps * len(dl))) if training_args.eval_steps < 1 else int(training_args.eval_steps),  # Added eval_steps
            logging_steps=max(1, int(training_args.logging_steps * len(dl))) if training_args.logging_steps < 1 else int(training_args.logging_steps),
            save_strategy=training_args.save_strategy,
            save_steps=training_args.save_steps,
            # eval_steps=int(training_args.eval_steps * len(dl)) if training_args.eval_steps < 1 else training_args.eval_steps,  # Added eval_steps
            # logging_steps=int(training_args.logging_steps * len(dl)) if training_args.logging_steps < 1 else training_args.logging_steps,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            learning_rate=training_args.learning_rate,  # Added learning_rate
            weight_decay=training_args.weight_decay,  # Added weight_decay
            warmup_ratio=training_args.warmup_ratio,  # Added warmup_ratio
            lr_scheduler_type=training_args.lr_scheduler_type,  # Added lr_scheduler_type
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,  # Added gradient_accumulation_steps
            gradient_checkpointing=training_args.gradient_checkpointing,  # Added gradient_checkpointing
            dataloader_pin_memory=training_args.dataloader_pin_memory,  # Added dataloader_pin_memory
            dataloader_num_workers=training_args.dataloader_num_workers,  # Added dataloader_num_workers
            save_total_limit=training_args.save_total_limit,  # Added save_total_limit
            load_best_model_at_end=training_args.load_best_model_at_end,
            report_to=training_args.report_to,
            remove_unused_columns=training_args.remove_unused_columns,  # Added remove_unused_columns
            seed=training_args.seed,  # Added seed
        ),
        train_dataset=dl.dataset,
        eval_dataset=dl.dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # Start training
    trainer.train()
    print_info("Training complete!")

    # Save the final model
    safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
    print_info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
