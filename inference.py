from tqdm import tqdm
import torch
import json
import os
import sys 
sys.path.append(os.getenv("ROOT", "/root/TracGPT-R3D"))
from src.model.load_model import load_model


def inference(model, tokenizer, dataloader, output_dir, num_beams=4, max_new_tokens=128, 
              do_sample=True, temperature=0.7, top_p=0.9, log_text_steps=5):
    """
    Fixed inference function with proper generation parameters
    """
    inference_file = os.path.join(output_dir, "inference_results.json")
    model.eval()
    
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
            batch = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(model.device)
                else:
                    batch[k] = v
            
            class_labels = inputs.get("class_labels", [])
            refs.extend(class_labels)
            for i in range(len(batch["input_ids"])):
                print("Input:", tokenizer.decode(batch["input_ids"][i], skip_special_tokens=False))
            print("Ground Truths:", class_labels)
            generated_ids = model.generate(
                images=batch["images"],
                input_ids=batch["input_ids"],   
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                # num_beams=num_beams,
                # do_sample=do_sample,
                # temperature=temperature,
                # top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # early_stopping=True,
                # repetition_penalty=1.1,
                # length_penalty=1.0
            )
            
            batch_preds = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True  
            )
            
            for j, pred in enumerate(batch_preds):
                input_text = tokenizer.decode(batch["input_ids"][j], skip_special_tokens=True)
                if pred.startswith(input_text):
                    pred = pred[len(input_text):].strip()
                
                print(f"Sample {i*len(batch_preds)+j}: {pred}")
                preds.append(pred)
                raw_preds.append(batch_preds[j])  
                
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

if __name__ == "__main__":
    from src.collators.standard_collator import StandardCollator
    from src.dataset.dataloader import load_data
    from metric import calculate_metric
    import yaml
    import shutil   
    # config_path="config/vit_llama_3B.yaml"
    config_path="config/vit_llama_3B_cot_full.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config=config["data"]
    device="cuda"
    model_config = config["model"]
    # pretrain_path="output/r8qyf5em/checkpoint-678"
    # pretrain_path="output/626yvz3p/checkpoint-120"
    pretrain_path="output/vlxjrszc/checkpoint-3000"
    tag=pretrain_path.split("/")[1]
    tokenizer,model=load_model(model_config,pretrain_path,lora=True)
    print("eos_token_id",tokenizer.eos_token_id)
    print("pad_token_id",tokenizer.pad_token_id)
    decoded = tokenizer.decode([tokenizer.pad_token_id], skip_special_tokens=False)
    print("decoded:", decoded)
    print(tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False))

    model.to(device)
    train_set, val_set, test_set= load_data(
        train_val_dir=data_config["train_val_dir"],
        test_dir=data_config["test_dir"],
        image_train_path=data_config["image_train_path"],
        image_test_path=data_config["image_test_path"],
        dataset=data_config["dataset"],
        train_sample=data_config["train_sample"],
        val_sample=data_config["val_sample"],
        test_sample=data_config["test_sample"],
        overfit_train=data_config["overfit_train"]
    )
    collator=StandardCollator(tokenizer=tokenizer)
    batch_size=2
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )   
    val_loader = torch.utils.data.DataLoader(    
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )   
    

    train_eval_output_dir=f"full_eval/{tag}_train"
    val_eval_output_dir=f"full_eval/{tag}_val"
    test_eval_output_dir=f"full_eval/{tag}_test"
    if os.path.exists(train_eval_output_dir):
        shutil.rmtree(train_eval_output_dir)  
    if os.path.exists(val_eval_output_dir):
        shutil.rmtree(val_eval_output_dir)  
    if os.path.exists(test_eval_output_dir):
        shutil.rmtree(test_eval_output_dir)
    os.makedirs(train_eval_output_dir,exist_ok=True)
    os.makedirs(val_eval_output_dir,exist_ok=True)
    os.makedirs(test_eval_output_dir,exist_ok=True)
    test_metrics = inference(
        model, tokenizer, test_loader, test_eval_output_dir,
    )
    # with open(f"{test_eval_output_dir}/inference_results.json", "r") as f:
    #     data=json.load(f)
    #     preds = data["predictions"]
    #     labels = data["references"]
    #     calculate_metric(preds, labels)