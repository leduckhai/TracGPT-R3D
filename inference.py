from tqdm import tqdm
import torch
import json
import os
import sys 
sys.path.append(os.getenv("ROOT", "/root/TracGPT-R3D"))
from src.model.load_model import load_model


def inference(model,tokenizer, dataloader,output_dir,num_beams=1, max_new_tokens=128, log_text_steps=5):
    inference_file = os.path.join(output_dir, "inference_results.json")
    model.eval()
    tokenizer = tokenizer
    preds = []
    refs = [] 
    for i, inputs in enumerate(tqdm(dataloader, desc="Generating")):
        batch = {k: v.to(model.device) for k, v in inputs.items() 
                if isinstance(v, torch.Tensor)}
        
        class_labels = inputs.get("class_labels", [])
        refs.extend(class_labels)  
        generated_ids = model.generate(
            images=batch["images"],
            input_ids=batch["input_ids"],   
            attention_mask=batch["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,  
                eos_token_id=tokenizer.eos_token_id,  
        )
        
        batch_preds = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True  
        )
        for pred in batch_preds:
            print("pred", pred)
            preds.append(pred)   
    with open(inference_file, "w") as f:
        json.dump({
            "predictions": preds,
            "references": refs
        }, f, indent=4)
    print("Inference results saved to", inference_file)
    return 

if __name__ == "__main__":
    from src.collators.standard_collator import StandardCollator
    from src.dataset.dataloader import load_data
    import yaml
    import shutil   
    from torch.nn.functional import softmax
    config_path="config/vit_llama_3B.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config=config["data"]
    device="cuda"
    model_config = config["model"]
    pretrain_path="output/ux7px8hw/checkpoint-2706"
    tokenizer,model=load_model(model_config,pretrain_path,lora=True)
    train_set, val_set, test_set= load_data(
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
    collator=StandardCollator(tokenizer=tokenizer,mode="test")
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
    # inference_file =  "inference_results.json"
    model.to(device)
    train_eval_output_dir="full_eval/ux7px8hw_train"
    val_eval_output_dir="full_eval/ux7px8hw_val"
    test_eval_output_dir="full_eval/ux7px8hw_test"
    if os.path.exists(train_eval_output_dir):
        shutil.rmtree(train_eval_output_dir)  
    if os.path.exists(val_eval_output_dir):
        shutil.rmtree(val_eval_output_dir)  
    if os.path.exists(test_eval_output_dir):
        shutil.rmtree(test_eval_output_dir)
    os.makedirs(train_eval_output_dir,exist_ok=True)
    os.makedirs(val_eval_output_dir,exist_ok=True)
    os.makedirs(test_eval_output_dir,exist_ok=True)
    print("Starting inference on train set")
    train_metrics=inference(model,tokenizer,train_loader,train_eval_output_dir)
    print("starting inference on test set")
    test_metrics=inference(model,tokenizer,test_loader,test_eval_output_dir)
