import sys 
sys.path.append("/root/TracGPT-R3D")
import torch
from transformers import AutoTokenizer
import os
from model.trac_llava import TracLlavaForCausalLM,TracConfig
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM
def load_model(config, pretrained_path=None, lora=False):
    base_model_name=config["config"]["language_model"]["name"]
    print("base_model_name",base_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if config["name"] == "vit_llama":
        custom_config = config["config"]
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model_name = custom_config["language_model"]["name"]
        print("Loading base model:", base_model_name)
        new_tokens = ["<image>", "<PAD>"]
        tokenizer.add_tokens(new_tokens, special_tokens=True)
        tokenizer.pad_token = "<PAD>"
        print(f"Set pad_token : {tokenizer.pad_token}")
        if not pretrained_path:
            print("Loading TracLlamaForCausalLM")
            base_model_name = custom_config["language_model"]["name"]
            config=TracConfig(custom_config)
            model = TracLlavaForCausalLM(config,tokenizer=tokenizer)
            return tokenizer, model
        else:
            print("Loading pretrained model from:", pretrained_path)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            config=TracConfig.from_pretrained(pretrained_path)
            if lora:
                print("Loading LoRA weights")
                
                from peft import PeftModel
                model = TracLlavaForCausalLM(config=config,tokenizer=tokenizer)
                projector_path = pretrained_path + "/projector.pth"
                lora_path= pretrained_path + "/lora"
               
                # base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
                # diff = (model.lm_model.model.layers[0].self_attn.q_proj.weight - 
                #         base_model.model.layers[0].self_attn.q_proj.weight).abs().sum()
                # print("Sum of changes Before merge dif:", diff.item())
                model.lm_model = PeftModel.from_pretrained(model.lm_model, lora_path)
                model.lm_model = model.lm_model.merge_and_unload()
                if os.path.exists(projector_path):
                    model.load_projector_weight(projector_path)
                    print(f"Loaded projector weights from {projector_path}")
                # merged_model = model.lm_model

                # diff = (merged_model.model.layers[0].self_attn.q_proj.weight - 
                #         base_model.model.layers[0].self_attn.q_proj.weight).abs().sum()
                # print("Sum of changes:", diff.item())
                return tokenizer, model
        
            else:
                print("Loading full model weights")
                model = TracLlavaForCausalLM.from_pretrained(pretrained_path, config=config)
                model.load_projector_weight(projector_path)

            return tokenizer, model
if __name__ == "__main__":
    import os
    import yaml
    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    config_path="config/vit_llama.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config= config["model"]
    pretrain_path="output/qs2tahbf/checkpoint-4"
    assert os.path.exists(pretrain_path), f"Pretrained path {pretrain_path} does not exist."

    tokenizer, model = load_model(model_config, pretrained_path=pretrain_path, lora=True)
    print("Model and tokenizer loaded successfully.")

    data_config = config["data"]
    device= "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
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
    collator = load_collator(config["general"]["collator"],tokenizer=tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    for i, batch in enumerate(val_loader):
        if i >= 1:
            break
        images = batch["images"].to(device)
        print("images shape", images.shape)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        full_texts = batch["full_texts"]
        labels = batch["labels"].to(device)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"   
        with torch.inference_mode():
            prompt_texts = batch["full_texts"]   
            device="cuda"
            enc = tokenizer(
                prompt_texts,
                padding=True,
                truncation=True,
                max_length=collator.max_length,
                return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                images=images,
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=50,
            )
            text=tokenizer.batch_decode(outputs, skip_special_tokens=True)

            print("Generated text:", text)
            # for i, t in enumerate(text):
            #     print(f" {i} Generated text:", t)
            # print("Full texts:", full_texts)
    
    