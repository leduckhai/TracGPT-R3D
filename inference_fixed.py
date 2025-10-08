#!/usr/bin/env python3
"""
Fixed inference script with all corrections applied
"""
import os
import sys
import torch
import yaml
import json
import shutil
from tqdm import tqdm

# Add project root to path
sys.path.append("/root/TracGPT-R3D")

from src.model.load_model import load_model
from src.dataset.dataloader import load_data
from src.collators.standard_collator import StandardCollator

def inference_fixed(model, tokenizer, dataloader, output_dir, 
                   num_beams=4, max_new_tokens=128, do_sample=True, 
                   temperature=0.7, top_p=0.9):
    """
    Fixed inference function with all corrections
    """
    inference_file = os.path.join(output_dir, "inference_results.json")
    model.eval()
    
    # Set consistent tokenizer state
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    preds = []
    refs = []
    raw_preds = []
    
    print(f"Starting inference with parameters:")
    print(f"  - num_beams: {num_beams}")
    print(f"  - max_new_tokens: {max_new_tokens}")
    print(f"  - do_sample: {do_sample}")
    print(f"  - temperature: {temperature}")
    print(f"  - top_p: {top_p}")
    
    with torch.inference_mode():
        for i, inputs in enumerate(tqdm(dataloader, desc="Generating")):
            # Proper device handling
            batch = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)
                else:
                    batch[k] = v
            
            class_labels = inputs.get("class_labels", [])
            refs.extend(class_labels)
            
            # Use proper generation parameters
            generated_ids = model.generate(
                images=batch["images"],
                input_ids=batch["input_ids"],   
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            
            # Better text decoding
            batch_preds = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True  
            )
            
            # Process predictions
            for j, pred in enumerate(batch_preds):
                # Remove input prompt from generated text
                input_text = tokenizer.decode(batch["input_ids"][j], skip_special_tokens=True)
                if pred.startswith(input_text):
                    pred = pred[len(input_text):].strip()
                
                print(f"Sample {i*len(batch_preds)+j}: {pred}")
                preds.append(pred)
                raw_preds.append(batch_preds[j])
                
    # Save comprehensive results
    results = {
        "predictions": preds,
        "raw_predictions": raw_preds,
        "references": refs,
        "generation_params": {
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p
        }
    }
    
    with open(inference_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Inference results saved to {inference_file}")
    return results

def main():
    """Main inference function"""
    print("TracGPT Fixed Inference Script")
    print("=" * 50)
    
    # Configuration
    config_path = "config/vit_llama_3B.yaml"
    pretrained_path = "output/hd4nazs3/checkpoint-2706"  # Update this path
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    model_config = config["model"]
    
    print(f"Loading model from: {pretrained_path}")
    print(f"Using LoRA: True")
    
    # Load model and tokenizer
    tokenizer, model = load_model(model_config, pretrained_path=pretrained_path, lora=True)
    if model is None:
        print("✗ Failed to load model")
        return
    
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("✓ Model loaded and moved to device")
    
    # Load data
    print("Loading data...")
    train_set, val_set, test_set = load_data(
        train_val_dir=data_config["train_val_dir"],
        test_dir=data_config["test_dir"],
        image_train_path=data_config["image_train_path"],
        image_test_path=data_config["image_test_path"],
        dataset=data_config["dataset"],
        train_sample=data_config["train_sample"],
        val_sample=data_config["val_sample"],
        test_sample=data_config["test_sample"],
        dataset_config=data_config["dataset_config"]
    )
    
    print(f"✓ Data loaded - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # Create collators
    train_collator = StandardCollator(tokenizer=tokenizer, mode="train")
    test_collator = StandardCollator(tokenizer=tokenizer, mode="test")
    
    # Create dataloaders
    batch_size = 2
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, collate_fn=train_collator
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=test_collator
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=test_collator
    )
    
    # Create output directories
    output_dirs = {
        "train": "fixed_eval/train",
        "val": "fixed_eval/val", 
        "test": "fixed_eval/test"
    }
    
    for name, output_dir in output_dirs.items():
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
    
    # Run inference
    print("\n" + "=" * 50)
    print("RUNNING INFERENCE")
    print("=" * 50)
    
    # Test on validation set first
    print("Running inference on validation set...")
    val_results = inference_fixed(
        model, tokenizer, val_loader, output_dirs["val"],
        num_beams=4, max_new_tokens=128, do_sample=True,
        temperature=0.7, top_p=0.9
    )
    
    # Test on test set
    print("Running inference on test set...")
    test_results = inference_fixed(
        model, tokenizer, test_loader, output_dirs["test"],
        num_beams=4, max_new_tokens=128, do_sample=True,
        temperature=0.7, top_p=0.9
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)
    print(f"Validation samples: {len(val_results['predictions'])}")
    print(f"Test samples: {len(test_results['predictions'])}")
    print(f"Results saved to: fixed_eval/")
    
    # Show some example predictions
    print("\nExample predictions:")
    for i, (pred, ref) in enumerate(zip(test_results['predictions'][:3], test_results['references'][:3])):
        print(f"  {i+1}. Predicted: {pred}")
        print(f"     Expected: {ref}")
        print()

if __name__ == "__main__":
    main()

