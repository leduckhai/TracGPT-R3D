from transformers import Trainer
import torch 
from tqdm import tqdm
import time
from collections import defaultdict
import os
import json 
import shutil
class StandardTrainer(Trainer):
    def __init__(self, tracker, *args, **kwargs):
        self.tracker = tracker
        self.eval_print_interval = 2
        super().__init__(*args, **kwargs)
        self.eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
        with open(self.eval_file, "w") as f:
            f.write("Evaluation\n")

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        batch = {
            'images': inputs['images'].to(model.device, non_blocking=True),
            'input_ids': inputs['input_ids'].to(model.device, non_blocking=True),
            'attention_mask': inputs['attention_mask'].to(model.device, non_blocking=True),
            'labels': inputs['labels'].to(model.device, non_blocking=True)
        }
 
        output=model(**batch)
        loss=output.loss
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch

        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        print("Loss", loss.item())
        if self.state.global_step % self.args.logging_steps == 0:
            self.tracker.log({
                "train/loss": loss.item(),
                "train/mem_alloc": torch.cuda.memory_allocated()/1e9  # Monitor memory
            }, step=self.state.global_step)
        
        return loss.detach()  
        
        
    def evaluate(self,  ignore_keys=None, max_new_tokens=128, num_beams=1):
        self.model.eval()
        torch.cuda.empty_cache()
        model=self.model 
        data_loader = self.get_eval_dataloader()
        losses=[]
        tokenizer=self.tokenizer
        preds = []  
        refs = []
        steps=[]
        with torch.no_grad(): 
            for i, inputs in enumerate(tqdm(data_loader, desc="Evaluation")):
              
                batch = {
                    'images': inputs['images'].to(model.device, non_blocking=True),
                    'input_ids': inputs['input_ids'].to(model.device, non_blocking=True),
                    'attention_mask': inputs['attention_mask'].to(model.device, non_blocking=True),
                    'labels': inputs['labels'].to(model.device, non_blocking=True)
                }
            
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = model(**batch)
                    loss = outputs.loss
                if i%self.eval_print_interval == 0:
                    print(f"Generate text output on evaluation")
                    generated_ids = self.model.generate(
                    images=inputs['images'].to(model.device, non_blocking=True),
                    input_ids=inputs['input_ids'].to(model.device, non_blocking=True),
                    attention_mask=inputs['attention_mask'].to(model.device, non_blocking=True),
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                    batch_preds = tokenizer.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True  
                    )
                    print("full text", inputs["full_texts"])
                    refs.extend(inputs["class_labels"])  
                    preds.extend(batch_preds)
                    steps.append(self.state.global_step)
                    for i,pred in enumerate(batch_preds):
                        print("Gt", inputs["class_labels"])
                        print("Pred", pred)
                losses.append(loss.item())
                del batch, outputs
                if self.state.global_step % self.args.logging_steps == 0:
                    self.tracker.log({
                        "eval/loss": loss.item(),
                        "eval/mem_alloc": torch.cuda.memory_allocated()/1e9  
                    }, step=self.state.global_step)
        with open(self.eval_file, "a") as f:
            f.write(
                f"Step {self.state.global_step}\n"
                f"refs: {refs}\n"
                f"preds: {preds}\n"
                f"steps: {steps}\n\n"
            )
        return {
            "eval_loss": sum(losses)/len(losses)
        }
    def inference(self, eval_dataset=None, ignore_keys=None, max_new_tokens=128, num_beams=1, log_text_steps=5):
        output_dir = self.args.output_dir
        inference_file = os.path.join(output_dir, "inference_results.json")
        print("Inference file:", inference_file)
        self.model.eval()
        tokenizer = self.tokenizer
        preds = []
        refs = [] 
        raw_preds = []
        torch.cuda.empty_cache()     
        dataloader = self.get_eval_dataloader(eval_dataset)
        with torch.inference_mode():
            for i, inputs in enumerate(tqdm(dataloader, desc="Inference")):      
                full_texts = inputs.get("full_texts", [])
                class_labels = inputs.get("class_labels", [])
                images=inputs.get("images", []).to(self.model.device)  
                input_ids=inputs.get("input_ids", []).to(self.model.device)
                attention_mask=inputs.get("attention_mask", []).to(self.model.device)            
                refs.extend(class_labels)  
   
                generated_ids = self.model.generate(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    # num_beams=num_beams,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                batch_preds = tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True  
                )
                for pred in batch_preds:
                    print("Pred", pred)
                    preds.append(pred)
                  
        
        metrics = {"num_samples": len(preds)}
        with open(inference_file, "w") as f:
            json.dump({
                "predictions": preds,
                "references": refs
            }, f, indent=4)
            
        return metrics
    
    def save_model(self, output_dir=None):
        super().save_model(output_dir)
        projector_path = os.path.join(output_dir, "mm_projector.pth")
        
        
