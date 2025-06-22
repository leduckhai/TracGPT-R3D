import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return -1, 1, -1

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    """Check if current process is the main process"""
    return rank in [-1, 0]

def rank0_print(*args):
    """Print only from rank 0 process"""
    if is_main_process(int(os.environ.get('RANK', -1))):
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    model_type: Optional[str] = field(default=None, metadata={"help": "llama2, phi3"})

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    # image
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))

    # vision
    vision_tower: Optional[str] = field(default="vit3d")
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"})
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})

@dataclass
class DataArguments:
    data_root: str = field(default="./Data/data/", metadata={"help": "Root directory for all data."})

    # caption data
    cap_data_path: str = field(default="./Data/data/M3D_Cap_npy/M3D_Cap.json", metadata={"help": "Path to caption data."})

    # VQA data
    vqa_data_train_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_train.csv", metadata={"help": "Path to training VQA data."})
    vqa_data_val_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_val.csv", metadata={"help": "Path to validation VQA data."})
    vqa_data_test_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_test.csv", metadata={"help": "Path to testing VQA data."})

    vqa_yn_data_train_path: str = field(default="./Data/data/M3D-VQA/M3D_VQA_yn_train.csv", metadata={"help": "Path to training VQA Yes or No data."})

    # positioning & segmentation data
    seg_data_path: str = field(default="./Data/data/M3D_Seg_npy/", metadata={"help": "Path to segmentation data."})
    refseg_data_train_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg.csv", metadata={"help": "Path to refering segmentation data."})
    refseg_data_test_path: str = field(default="./Data/data/M3D_RefSeg_npy/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    
    # Distributed training settings
    ddp_backend: str = "nccl"
    ddp_timeout: int = 18000  # Increased timeout for stability
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25  # Optimize DDP communication
    
    optim: str = field(default="adamw_torch")

    # Training configuration
    bf16: bool = True
    output_dir: str = "./LaMed/output/LaMed-pretrain-distributed"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10
    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4  # Increased for distributed training
    report_to: str = "tensorboard"
    
    # Enable distributed training
    local_rank: int = field(default=-1)
    
    def __post_init__(self):
        # Don't override environment variables if they're already set
        if self.local_rank == -1 and 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
        super().__post_init__()

def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions

    labels = labels_ids[:, 1:]
    preds = pred_ids[:, :-1]

    labels_flatten = labels.reshape(-1)
    preds_flatten = preds.reshape(-1)
    valid_indices = np.where(labels_flatten != -100)
    filtered_preds = preds_flatten[valid_indices]
    filtered_labels = labels_flatten[valid_indices]
    acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)

    return {"accuracy": acc_score}

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ['mm_projector', 'embed_tokens']

        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        
        # Only save from main process
        if is_main_process(trainer.args.local_rank):
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self, seg_enable):
        self.seg_enable = seg_enable
        
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 32, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
            )
        else:
            images, input_ids, labels, attention_mask = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        return return_dict

def main():
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set local rank from distributed setup
    if local_rank != -1:
        training_args.local_rank = local_rank

    # Set random seeds for reproducibility across all processes
    transformers.set_seed(training_args.seed)

    rank0_print("="*20 + " Distributed Training Setup " + "="*20)
    rank0_print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
    
    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Define and add special tokens
    special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
    tokenizer.add_special_tokens(special_token)
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if 'llama3' in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    
    # Load model
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            # model = LamedLlamaForCausalLM.from_pretrained(...)
            print("Warning: LamedLlamaForCausalLM not imported. Using standard LlamaForCausalLM")
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            )
        elif 'phi3' in model_args.model_type:
            # model = TracPhi3ForCausalLM.from_pretrained(...)
            print("Warning: TracPhi3ForCausalLM not imported. Using standard LlamaForCausalLM")
            model = LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        )

    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Move model to GPU
    if local_rank != -1:
        model = model.to(f'cuda:{local_rank}')
    else:
        model = model.cuda()

    rank0_print("Model setup complete. Add dataset and trainer initialization as needed.")
    
    # Note: You would add your dataset and trainer setup here
    # The trainer will automatically handle DDP wrapping when local_rank is set
    
    # Example trainer setup (you'll need to add your actual dataset):
    # trainer = transformers.Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,  # Add your dataset here
    #     eval_dataset=eval_dataset,    # Add your dataset here
    #     tokenizer=tokenizer,
    #     data_collator=DataCollator(seg_enable=False),  # Adjust based on your needs
    #     compute_metrics=compute_metrics,
    #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    # )
    
    # Clean up distributed training when done
    if dist.is_initialized():
        cleanup_distributed()

if __name__ == "__main__":
    main()