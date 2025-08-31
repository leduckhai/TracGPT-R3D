from tqdm import tqdm
import torch
import json
import os
import sys 
sys.path.append(os.getenv("ROOT", "/root/TracGPT-R3D"))
from src.model.load_model import load_model

def extract_answer(text, tokenizer):
    print("Extracting answer from text:", text)
    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    
    answer_start = text.find("<Answer>") + len("<Answer>")
    eos_pos = text.find(tokenizer.eos_token)
    
    if answer_start >= 0 and eos_pos >= 0:
        answer = text[answer_start:eos_pos].strip()
    else:
        answer = ""  
    
    return answer

def infer_test_data(model,tokenizer, dataloader,output_dir,num_beams=1, max_new_tokens=128, log_text_steps=5):
    inference_file = os.path.join(output_dir, "inference_results.json")
    model.eval()
    tokenizer = tokenizer
    preds = []
    refs = [] 
    for i, inputs in enumerate(tqdm(dataloader, desc="Generating")):
            p_ids=inputs.get("p_ids", [])
            print("P_IDs in batch:", p_ids)
            batch = {k: v.to(model.device) for k, v in inputs.items() 
                    if isinstance(v, torch.Tensor)}
            
            full_texts = inputs.get("full_texts", [])
            class_labels = inputs.get("class_labels", [])
            refs.extend(class_labels)  
            generated_ids = model.generate_with_images(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,  
                    eos_token_id=tokenizer.eos_token_id,  
            )
            
            batch_preds = tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True  
            )
            processed_preds=[]
            for pred in batch_preds:
                # answer = extract_answer(pred, tokenizer)
                # processed_preds.append(answer)
                # print(f"Extracted answer: {answer}")
                print("pred", pred)
                preds.append(pred)
            
            
            del generated_ids, batch
            torch.cuda.empty_cache()
    
    metrics = {"num_samples": len(preds)}
    print("refs", refs,len(refs), "preds", len(preds))
   
    with open(inference_file, "w") as f:
        json.dump({
            "metrics": metrics,
            "predictions": preds,
            "references": refs if refs else None
        }, f, indent=4)
    print("Inference results saved to", inference_file)
    print("final metrics", metrics)
    return metrics

if __name__ == "__main__":
    from src.collators.standard_collator import StandardCollator
    from src.dataset.dataloader import load_data
    import yaml
    config_path="config/vit_llama_3B.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_config=config["data"]
    device="cuda"
    model_config = config["model"]
    pretrain_path="output/usmypbp0/checkpoint-1932"
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
    collator=StandardCollator(tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=2,
        shuffle=False,
        collate_fn=collator
    )
    inference_file =  "inference_results.json"
    refs=[]
    preds=[]
    model.to(device)
    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            
            for i, inputs in enumerate(tqdm(dataloader, desc="Inference")):
                                       
                    full_texts = inputs.get("full_texts", [])
                    class_labels = inputs.get("class_labels", [])
                    images=inputs.get("images", []).to(device)
                     
                    refs.extend(class_labels)  
                    
                    enc = tokenizer(
                        full_texts,
                        padding=True,
                        truncation=True,
                        max_length=300,
                        return_tensors="pt"
                    ).to(device)
                    generated_ids = model.generate(
                        images=images,
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        max_new_tokens=300,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    
                    batch_preds = tokenizer.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True  
                    )
                    processed_preds=[]
                    for pred in batch_preds:
                        print("Pred", pred)
                        preds.append(pred)
                    
                    # del generated_ids, batch
                    torch.cuda.empty_cache()
            
            metrics = {"num_samples": len(preds)}
            print("preds", len(preds))
            
            with open(inference_file, "w") as f:
                json.dump({
                    "predictions": preds,
                    "references": refs
                }, f, indent=4)
                