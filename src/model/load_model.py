import sys 
sys.path.append("/root/TracGPT-R3D")
from src.model.vision_encoder.rcnn import TrainableFasterRCNN
from src.model.language.llama import TracLlamaForCausalLM, TracLlamaConfig, prepare_multi_modal_input
import torch
from transformers import AutoTokenizer

def load_model(config,pretrained_path=None,lora=False):
    if config["name"]=="vit_llama":
        custom_config = config["config"]
        config= TracLlamaConfig(config=custom_config)
        base_model_name = custom_config["language_model"]["name"]
        print("Loading base model:", base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if pretrained_path:
            print("Loading pretrained model from:", pretrained_path)
            if lora:
                print("Loading LoRA weights")
                from peft import PeftModel
                base_model = TracLlamaForCausalLM(config=config,tokenizer=tokenizer)
                model = PeftModel.from_pretrained(base_model, pretrained_path)

                # Merge adapter weights (optional, converts to full model)
                # model = model.merge_and_unload()
                # model = TracLlamaForCausalLM.from_pretrained(pretrained_path, config=config)
           
            else:
                print("Loading full model weights")
                model = TracLlamaForCausalLM.from_pretrained(pretrained_path, config=config)
        else:
            print("Initializing new model with config:", config)
            model = TracLlamaForCausalLM(config=config,tokenizer=tokenizer)
        # model = TracLlamaForCausalLM(config)
        device= "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print("eos_token:", tokenizer.eos_token)
        print("pad_token:", tokenizer.pad_token)
        tokenizer.pad_token = tokenizer.eos_token 
        tokenizer.padding_side = "right"  
        return tokenizer, model
        
    else:
        raise ValueError(f"Model {config.vision_backbone} not found")
if __name__ == "__main__":
    import os
    import yaml
    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    config_path="config/vit_llama.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config= config["model"]
    pretrain_path="output/vmxnsx4m/checkpoint-523"
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
        # outputs = model(
        #     images=images,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels
        # )
        with torch.inference_mode():
            outputs = model.generate_with_images(
            images=images,
            input_ids=input_ids,
            max_new_tokens=50,
            attention_mask=attention_mask,
            temperature=0.7,
            top_p=0.9
        )
            # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            text=tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print("Generated text:", text)
            for i, t in enumerate(text):
                print(f" {i} Generated text:", t)
            print("Full texts:", full_texts)
    
    