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
                num_beams=num_beams,
                early_stopping=True, 
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
                answer = extract_answer(pred, tokenizer)
                processed_preds.append(answer)
                print(f"Extracted answer: {answer}")
                
                preds.append(answer)
            
            
            if i % log_text_steps == 0:
                
                print(f"Pred: {processed_preds}")
                if class_labels:
                        print(f"Ref: {class_labels}")
            
            del generated_ids, batch
            torch.cuda.empty_cache()
    
    metrics = {"num_samples": len(preds)}
    print("refs", refs,len(refs), "preds", len(preds))
    if refs and len(refs) == len(preds):
        matches = sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip())
        metrics["eval/precision"] = matches / len(preds)
        print("eval precision", metrics["eval/precision"])
        # tracker.log({"eval/precision": metrics["eval/precision"]}, step=.state.global_step)
    elif refs:
        print(f"Warning: {len(preds)} preds vs {len(refs)} refs - skipping precision calc")
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
    
    # img
    # Example usage
    # model, tokenizer, dataloader, output_dir should be defined
    # infer_test_data(model, tokenizer, dataloader, output_dir)
    pass  # Replace with actual model, tokenizer, dataloader, and output_dir initialization