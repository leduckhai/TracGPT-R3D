from transformers import Trainer
import torch 
from tqdm import tqdm
import time
from collections import defaultdict
import os
import json 

class StandardTrainer(Trainer):
    def __init__(self, tracker, *args, **kwargs):
        self.tracker = tracker
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        with torch.no_grad(): 
            batch = {
                'images': inputs['images'].to(model.device, non_blocking=True),
                'input_ids': inputs['input_ids'].to(model.device, non_blocking=True),
                'attention_mask': inputs['attention_mask'].to(model.device, non_blocking=True),
                'labels': inputs['labels'].to(model.device, non_blocking=True)
            }
        
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            outputs = model(**batch)
            loss = outputs.loss
        
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        
        if self.args.fp16:
            self.scaler.scale(loss).backward()  # For mixed precision
        else:
            loss.backward()
        
        if self.state.global_step % self.args.logging_steps == 0:
            self.tracker.log({
                "train/loss": loss.item(),
                "train/mem_alloc": torch.cuda.memory_allocated()/1e9  # Monitor memory
            }, step=self.state.global_step)
        
        del batch, outputs
        torch.cuda.empty_cache() 
        
        return loss.detach()  
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, max_new_tokens=128, num_beams=1):
        self.model.eval()
        torch.cuda.empty_cache() 
        
        # total_loss = 0.0
        # num_batches = 0
        
        # dataloader = self.get_eval_dataloader(eval_dataset)
        
        # for batch in tqdm(dataloader, desc="Evaluating"):
            
        #     with torch.no_grad():
        #         batch = {k: v.to(self.model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
        #         outputs = self.model(**batch)
        #         loss = outputs.loss
                
        #         total_loss += loss.item()
        #         num_batches += 1
                
        #         if self.state.global_step % self.args.logging_steps == 0:
        #             self.tracker.log({"eval/loss": loss.item()}, step=self.state.global_step)
        
        # eval_loss = total_loss / num_batches if num_batches > 0 else 0.0
        # print(f"Evaluation completed. Mean loss: {eval_loss:.4f}")
        refs = []
        dataloader = self.get_eval_dataloader(eval_dataset)
        tokenizer = self.tokenizer
        with torch.inference_mode():
            for i, inputs in enumerate(tqdm(dataloader, desc="EVAL GENERATING")):
                if i==3:
                    break
                p_ids=inputs.get("p_ids", [])
                print("P_IDs in batch:", p_ids)
                batch = {k: v.to(self.model.device) for k, v in inputs.items() 
                        if isinstance(v, torch.Tensor)}
                
                full_texts = inputs.get("full_texts", [])
                class_labels = inputs.get("class_labels", [])
                refs.extend(class_labels)  
                tokenizer.padding_side = "left"
                tokenizer.truncation_side = "left"   
                enc = tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    max_length=300,
                    return_tensors="pt"
                 ).to(self.model.device)
                generated_ids = self.model.generate(
                    images=batch["images"],
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                    # early_stopping=True, 
                    # pad_token_id=tokenizer.eos_token_id,  
                    #  eos_token_id=tokenizer.eos_token_id,  
                )
                
                batch_preds = tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True  
                )
                processed_preds=[]
                for pred in batch_preds:
                    print("RAW PRED EVAL", pred)
                    answer = extract_answer(pred, tokenizer)
                    processed_preds.append(answer)
                    print(f"Extracted answer: {answer}")
                  
                    # preds.append(answer)
                
                
                # if i % log_text_steps == 0:
                    
                #     print(f"Pred: {processed_preds}")
                #     if full_texts:
                #             print(f"Ref: {full_texts}")
        return {"eval_loss": 0.0}
    def inference(self, eval_dataset=None, ignore_keys=None, max_new_tokens=128, num_beams=1, log_text_steps=5):
        output_dir = self.args.output_dir
        inference_file = os.path.join(output_dir, "inference_results.json")
        self.model.eval()
        tokenizer = self.tokenizer
        preds = []
        refs = [] 

        torch.cuda.empty_cache()
        
        dataloader = self.get_eval_dataloader(eval_dataset)
        
        with torch.inference_mode():
            for i, inputs in enumerate(tqdm(dataloader, desc="Generating")):
                
                p_ids=inputs.get("p_ids", [])
                print("P_IDs in batch:", p_ids)
                batch = {k: v.to(self.model.device) for k, v in inputs.items() 
                        if isinstance(v, torch.Tensor)}
                
                full_texts = inputs.get("full_texts", [])
                class_labels = inputs.get("class_labels", [])
                refs.extend(class_labels)  
                tokenizer.padding_side = "left"
                tokenizer.truncation_side = "left"   
                enc = tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    max_length=300,
                    return_tensors="pt"
                 ).to(self.model.device)
                generated_ids = self.model.generate(
                    images=batch["images"],
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                    # early_stopping=True, 
                    # pad_token_id=tokenizer.eos_token_id,  
                    #  eos_token_id=tokenizer.eos_token_id,  
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
                    if full_texts:
                            print(f"Ref: {full_texts}")
                
                del generated_ids, batch
                torch.cuda.empty_cache()
        
        metrics = {"num_samples": len(preds)}
        print("refs", refs,len(refs), "preds", len(preds))
        if refs and len(refs) == len(preds):
            matches = sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip())
            metrics["eval/precision"] = matches / len(preds)
            if self.tracker:
                self.tracker.log({"eval/precision": metrics["eval/precision"]}, step=self.state.global_step)
        elif refs:
            print(f"Warning: {len(preds)} preds vs {len(refs)} refs - skipping precision calc")
        with open(inference_file, "w") as f:
            json.dump({
                "metrics": metrics,
                "predictions": preds,
                "references": refs if refs else None
            }, f, indent=4)
            
        return metrics
def extract_answer(text, tokenizer):
    print("Extracting answer from text:", text)
    if not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    
    answer_start = text.find("<answer>") + len("<answer>")
    eos_pos = text.find(tokenizer.eos_token)
    
    if answer_start >= 0 and eos_pos >= 0:
        answer = text[answer_start:eos_pos].strip()
    else:
        answer = ""  
    
    return answer

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    text = "<Question> How substantial is this illness? <Answer> Status (Non-Dementia) <|end_of_text|>"
    # tokenizer.eos_token = "</s>"  # Set your tokenizer's EOS token
    answer = extract_answer(text, tokenizer)
    print(answer)  # Output: "Paris"