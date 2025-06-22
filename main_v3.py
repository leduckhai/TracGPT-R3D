#!/usr/bin/env python3
"""
Complete training pipeline for TracPhi3 multimodal model
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoConfig, 
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import logging

# Import your model classes (assuming they're in a separate file)
from trac_phi3_model import (
    TracPhi3Config, TracPhi3ForCausalLM, 
    MultimodalConfig, MultimodalProcessor
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    
    # Vision model arguments
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_vision_model: Optional[str] = field(default=None)
    
    # Vision configuration
    image_size: int = field(default=336)
    patch_size: int = field(default=14)
    image_channel: int = field(default=3)
    mm_hidden_size: int = field(default=2560)
    
    # Vision processing
    vision_select_layer: int = field(default=-1)
    vision_select_feature: str = field(default="patch")
    
    # Projector configuration
    proj_layer_type: str = field(default="linear")
    proj_layer_num: int = field(default=2)
    proj_pooling_type: str = field(default="avg")
    proj_pooling_size: int = field(default=2)
    
    # 3D bbox configuration
    bbox3d_module: Optional[str] = field(default="simple_bbox_head")
    bbox3d_token_id: Optional[int] = field(default=None)


@dataclass 
class DataArguments:
    """Arguments for data loading and processing"""
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(TrainingArguments):
    """Extended training arguments for multimodal training"""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


class MultimodalDataset(Dataset):
    """Dataset for multimodal training with text, images, and 3D bboxes"""
    
    def __init__(
        self, 
        data_path: str,
        tokenizer: AutoTokenizer,
        image_folder: str,
        image_processor,
        model_args: ModelArguments,
        data_args: DataArguments
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.model_args = model_args
        self.data_args = data_args
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return self._process_sample(sample)
    
    def _process_sample(self, sample):
        """Process a single training sample"""
        # Extract components
        conversations = sample.get('conversations', [])
        image_path = sample.get('image', None)
        bbox_3d = sample.get('bbox_3d', None)
        
        # Process conversation
        text_input = self._format_conversation(conversations)
        
        # Tokenize
        tokenized = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            max_length=self.data_args.model_max_length,
            truncation=True,
        )
        
        input_ids = tokenized["input_ids"][0]
        labels = input_ids.clone()
        
        # Process image
        image = None
        if image_path and self.image_folder:
            try:
                full_image_path = os.path.join(self.image_folder, image_path)
                image = Image.open(full_image_path).convert('RGB')
                image = self.image_processor(image)
                if isinstance(image, dict):
                    image = image['pixel_values'][0]
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                image = None
        
        # Process 3D bboxes
        bbox_gts = None
        bbox_masks = None
        if bbox_3d:
            bbox_gts, bbox_masks = self._process_bbox_3d(bbox_3d)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'images': image,
            'bbox_gts': bbox_gts,
            'bbox_masks': bbox_masks,
        }
    
    def _format_conversation(self, conversations):
        """Format conversation for training"""
        formatted_text = ""
        for turn in conversations:
            role = turn.get('from', 'user')
            value = turn.get('value', '')
            
            if role == 'human':
                formatted_text += f"Human: {value}\n"
            elif role == 'gpt':
                formatted_text += f"Assistant: {value}\n"
        
        return formatted_text
    
    def _process_bbox_3d(self, bbox_3d):
        """Process 3D bounding box annotations"""
        if not bbox_3d:
            return None, None
        
        # Convert to tensor format
        # Assuming bbox_3d is list of [x, y, z, w, h, d] format
        max_boxes = 10  # Maximum number of boxes per sample
        
        if isinstance(bbox_3d, list) and len(bbox_3d) > 0:
            # Handle multiple boxes
            num_boxes = min(len(bbox_3d), max_boxes)
            bbox_tensor = torch.zeros(max_boxes, 6)  # 6D bounding box
            mask_tensor = torch.zeros(max_boxes, dtype=torch.bool)
            
            for i in range(num_boxes):
                bbox_tensor[i] = torch.tensor(bbox_3d[i][:6])
                mask_tensor[i] = True
            
            return bbox_tensor, mask_tensor
        
        return None, None


class CustomDataCollator(DataCollatorForSeq2Seq):
    """Custom data collator for multimodal data"""
    
    def __init__(self, tokenizer, model=None, **kwargs):
        super().__init__(tokenizer, model, **kwargs)
    
    def __call__(self, features):
        # Separate multimodal features
        batch = {}
        
        # Text features
        text_features = []
        for feature in features:
            text_features.append({
                'input_ids': feature['input_ids'],
                'labels': feature['labels'],
            })
        
        # Use parent collator for text
        text_batch = super().__call__(text_features)
        batch.update(text_batch)
        
        # Handle images
        images = [f['images'] for f in features if f['images'] is not None]
        if images:
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = None
        
        # Handle 3D bboxes
        bbox_gts = [f['bbox_gts'] for f in features if f['bbox_gts'] is not None]
        bbox_masks = [f['bbox_masks'] for f in features if f['bbox_masks'] is not None]
        
        if bbox_gts:
            batch['bbox_gts'] = torch.stack(bbox_gts)
            batch['bbox_masks'] = torch.stack(bbox_masks)
        else:
            batch['bbox_gts'] = None
            batch['bbox_masks'] = None
        
        return batch


def setup_tokenizer(model_args: ModelArguments):
    """Setup tokenizer with special tokens"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<pad>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }
    
    # Add multimodal tokens
    special_tokens["additional_special_tokens"] = [
        "<image>", "<bbox>", "<image_newline>"
    ]
    
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    
    # Set image token attributes
    tokenizer.image_token = "<image>"
    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    tokenizer.bbox_token = "<bbox>"
    tokenizer.bbox_token_id = tokenizer.convert_tokens_to_ids("<bbox>")
    tokenizer.image_newline_token = "<image_newline>"
    tokenizer.image_newline_token_id = tokenizer.convert_tokens_to_ids("<image_newline>")
    
    return tokenizer, num_new_tokens


def setup_model(model_args: ModelArguments, tokenizer, num_new_tokens: int = 0):
    """Setup and initialize the TracPhi3 model"""
    
    # Create config
    config = TracPhi3Config.from_pretrained(model_args.model_name_or_path)
    
    # Update config with multimodal parameters
    multimodal_params = {
        'vision_tower': model_args.vision_tower,
        'mm_projector_type': model_args.mm_projector_type,
        'image_size': model_args.image_size,
        'patch_size': model_args.patch_size,
        'image_channel': model_args.image_channel,
        'mm_hidden_size': model_args.mm_hidden_size,
        'vision_select_layer': model_args.vision_select_layer,
        'vision_select_feature': model_args.vision_select_feature,
        'proj_layer_type': model_args.proj_layer_type,
        'proj_layer_num': model_args.proj_layer_num,
        'proj_pooling_type': model_args.proj_pooling_type,
        'proj_pooling_size': model_args.proj_pooling_size,
        'bbox3d_module': model_args.bbox3d_module,
        'bbox3d_token_id': tokenizer.bbox_token_id,
    }
    
    # Update config
    for key, value in multimodal_params.items():
        setattr(config, key, value)
    
    # Create model
    logger.info("Creating TracPhi3ForCausalLM model...")
    model = TracPhi3ForCausalLM(config)
    
    # Resize token embeddings if new tokens were added
    if num_new_tokens > 0:
        logger.info(f"Resizing token embeddings for {num_new_tokens} new tokens")
        model.resize_token_embeddings(len(tokenizer))
    
    # Initialize multimodal components
    logger.info("Initializing multimodal components...")
    model.initialize_multimodal_components(model_args, tokenizer)
    
    # Setup training parameters
    if model_args.freeze_backbone:
        logger.info("Freezing backbone parameters")
        for param in model.model.parameters():
            param.requires_grad = False
    
    if model_args.tune_mm_mlp_adapter:
        logger.info("Enabling gradients for multimodal MLP adapter")
        for p in model.model.vision_encoder.mm_projector.parameters():
            p.requires_grad = True
    
    return model


def setup_image_processor(model_args: ModelArguments):
    """Setup image processor"""
    try:
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(
            model_args.vision_tower,
            size=model_args.image_size
        )
    except:
        # Fallback image processor
        class SimpleImageProcessor:
            def __init__(self, size=336):
                self.size = size
            
            def __call__(self, image):
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                return transform(image)
        
        image_processor = SimpleImageProcessor(model_args.image_size)
    
    return image_processor


class TracPhi3Trainer(Trainer):
    """Custom trainer for TracPhi3 multimodal training"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation including 3D bbox loss"""
        
        # Forward pass
        outputs = model(**inputs)
        
        # Get language modeling loss
        loss = outputs.loss
        
        # Add 3D bbox loss if available
        if hasattr(outputs, 'bbox_3d_loss'):
            bbox_loss = outputs.bbox_3d_loss
            loss = loss + 0.1 * bbox_loss  # Weight the bbox loss
            
            # Log losses separately
            self.log({
                "train/lm_loss": outputs.loss.item(),
                "train/bbox_loss": bbox_loss.item(),
                "train/total_loss": loss.item(),
            })
        
        return (loss, outputs) if return_outputs else loss


def main():
    """Main training function"""
    
    # Parse arguments (you can use argparse or hydra for this)
    model_args = ModelArguments()
    data_args = DataArguments(
        data_path="path/to/your/training_data.json",
        image_folder="path/to/your/images/"
    )
    training_args = TrainingArguments(
        output_dir="./trac_phi3_output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        dataloader_pin_memory=False,
        fp16=True,
        remove_unused_columns=False,
    )
    
    # Setup tokenizer
    logger.info("Setting up tokenizer...")
    tokenizer, num_new_tokens = setup_tokenizer(model_args)
    
    # Setup image processor
    logger.info("Setting up image processor...")
    image_processor = setup_image_processor(model_args)
    
    # Setup model
    logger.info("Setting up model...")
    model = setup_model(model_args, tokenizer, num_new_tokens)
    
    # Setup dataset
    logger.info("Setting up dataset...")
    train_dataset = MultimodalDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        image_folder=data_args.image_folder,
        image_processor=image_processor,
        model_args=model_args,
        data_args=data_args
    )
    
    # Setup data collator
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Setup trainer
    logger.info("Setting up trainer...")
    trainer = TracPhi3Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()


# Example usage scripts:

def example_inference():
    """Example inference with trained model"""
    
    # Load model and tokenizer
    model_path = "./trac_phi3_output"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TracPhi3ForCausalLM.from_pretrained(model_path)
    
    # Prepare input
    text = "Human: What objects do you see in this image? <image>\nAssistant:"
    image = Image.open("example.jpg").convert('RGB')
    
    # Process image (you'd need the same image processor used in training)
    image_processor = setup_image_processor(ModelArguments())
    processed_image = image_processor(image).unsqueeze(0)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            images=processed_image,
            max_length=512,
            temperature=0.7,
            do_sample=True,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


def example_bbox_inference():
    """Example inference with 3D bbox prediction"""
    
    model_path = "./trac_phi3_output"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TracPhi3ForCausalLM.from_pretrained(model_path)
    
    text = "Human: Where is the car in this image? <image> <bbox>\nAssistant:"
    image = Image.open("street_scene.jpg").convert('RGB')
    
    image_processor = setup_image_processor(ModelArguments())
    processed_image = image_processor(image).unsqueeze(0)
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        # Generate with bbox prediction enabled
        sequences, bbox_predictions = model.generate(
            input_ids=inputs["input_ids"],
            images=processed_image,
            bbox3d_enable=True,
            max_length=512,
            return_dict_in_generate=True,
        )
    
    response = tokenizer.decode(sequences[0], skip_special_tokens=True)
    print("Response:", response)
    print("3D BBox prediction:", bbox_predictions)