#!/usr/bin/env python3
"""
Debug script to validate inference fixes
"""
import os
import sys
import torch
import yaml
import json
from transformers import AutoTokenizer

# Add project root to path
sys.path.append("/root/TracGPT-R3D")

from src.model.load_model import load_model
from src.dataset.dataloader import load_data
from src.collators.standard_collator import StandardCollator

def validate_model_loading(config_path, pretrained_path, lora=True):
    """Validate that model loading works correctly"""
    print("=" * 50)
    print("VALIDATING MODEL LOADING")
    print("=" * 50)
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        model_config = config["model"]
        print(f"Loading model from: {pretrained_path}")
        print(f"Using LoRA: {lora}")
        
        tokenizer, model = load_model(model_config, pretrained_path=pretrained_path, lora=lora)
        
        print("✓ Model loaded successfully")
        print(f"✓ Tokenizer vocab size: {len(tokenizer)}")
        print(f"✓ Model device: {next(model.parameters()).device}")
        print(f"✓ Model dtype: {next(model.parameters()).dtype}")
        
        # Check if projector weights are loaded
        if hasattr(model, 'mm_projector'):
            print("✓ Projector found")
            projector_params = sum(p.numel() for p in model.mm_projector.parameters())
            print(f"✓ Projector parameters: {projector_params:,}")
        else:
            print("⚠ Projector not found")
        
        # Check LoRA adapters
        if lora and hasattr(model, 'peft_config'):
            print("✓ LoRA adapters loaded")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✓ Trainable parameters: {trainable_params:,}")
        else:
            print("⚠ LoRA adapters not found")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None, None

def validate_data_processing(config_path, tokenizer):
    """Validate data processing consistency"""
    print("\n" + "=" * 50)
    print("VALIDATING DATA PROCESSING")
    print("=" * 50)
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        data_config = config["data"]
        
        # Load data
        train_set, val_set, test_set = load_data(
            train_val_dir=data_config["train_val_dir"],
            test_dir=data_config["test_dir"],
            image_train_path=data_config["image_train_path"],
            image_test_path=data_config["image_test_path"],
            dataset=data_config["dataset"],
            train_sample=5,  # Small sample for debugging
            val_sample=5,
            test_sample=5,
            dataset_config=data_config["dataset_config"]
        )
        
        print(f"✓ Data loaded - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        
        # Test collators
        train_collator = StandardCollator(tokenizer=tokenizer, mode="train")
        test_collator = StandardCollator(tokenizer=tokenizer, mode="test")
        
        # Test with a small batch
        train_batch = [train_set[i] for i in range(min(2, len(train_set)))]
        test_batch = [test_set[i] for i in range(min(2, len(test_set)))]
        
        train_processed = train_collator(train_batch)
        test_processed = test_collator(test_batch)
        
        print("✓ Collators work correctly")
        print(f"✓ Train batch shape: {train_processed['input_ids'].shape}")
        print(f"✓ Test batch shape: {test_processed['input_ids'].shape}")
        
        # Check tokenization consistency
        train_text = train_processed['full_texts'][0]
        test_text = test_processed['full_texts'][0]
        
        print(f"✓ Train text: {train_text[:100]}...")
        print(f"✓ Test text: {test_text[:100]}...")
        
        return train_set, val_set, test_set, train_collator, test_collator
        
    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        return None, None, None, None, None

def validate_inference(model, tokenizer, test_collator, test_set):
    """Validate inference generation"""
    print("\n" + "=" * 50)
    print("VALIDATING INFERENCE")
    print("=" * 50)
    
    try:
        model.eval()
        device = next(model.parameters()).device
        
        # Create test dataloader
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            collate_fn=test_collator
        )
        
        print("✓ Test dataloader created")
        
        # Test generation
        with torch.inference_mode():
            for i, batch in enumerate(test_loader):
                if i >= 2:  # Test only 2 samples
                    break
                
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                print(f"\n--- Sample {i+1} ---")
                print(f"Question: {batch['question_texts'][0]}")
                print(f"Expected: {batch['class_labels'][0]}")
                
                # Generate
                generated_ids = model.generate(
                    images=batch["images"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=50,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True
                )
                
                # Decode
                generated_text = tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                
                # Remove input prompt
                input_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
                if generated_text.startswith(input_text):
                    generated_text = generated_text[len(input_text):].strip()
                
                print(f"Generated: {generated_text}")
                
                if generated_text.strip():
                    print("✓ Generation successful")
                else:
                    print("⚠ Empty generation")
        
        print("✓ Inference validation completed")
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debugging function"""
    print("TracGPT Inference Debug Script")
    print("=" * 50)
    
    # Configuration
    config_path = "config/vit_llama_3B.yaml"
    pretrained_path = "output/hd4nazs3/checkpoint-2706"  # Update this path
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return
    
    if not os.path.exists(pretrained_path):
        print(f"✗ Pretrained path not found: {pretrained_path}")
        print("Please update the pretrained_path variable in this script")
        return
    
    # Step 1: Validate model loading
    tokenizer, model = validate_model_loading(config_path, pretrained_path, lora=True)
    if model is None:
        print("✗ Cannot proceed without model")
        return
    
    # Step 2: Validate data processing
    train_set, val_set, test_set, train_collator, test_collator = validate_data_processing(
        config_path, tokenizer
    )
    if test_collator is None:
        print("✗ Cannot proceed without data processing")
        return
    
    # Step 3: Validate inference
    success = validate_inference(model, tokenizer, test_collator, test_set)
    
    print("\n" + "=" * 50)
    if success:
        print("✓ ALL VALIDATIONS PASSED - Inference should work correctly!")
    else:
        print("✗ SOME VALIDATIONS FAILED - Check the errors above")
    print("=" * 50)

if __name__ == "__main__":
    main()

