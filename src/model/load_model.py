import sys 
sys.path.append("/root/TracGPT-R3D")
from src.model.vision_encoder.rcnn import TrainableFasterRCNN
# from src.model.language.llama import TracLlamaForCausalLM, TracLlamaConfig, prepare_multi_modal_input
import torch
from transformers import AutoTokenizer
from src.model.llava_origin_v2 import TracLlavaForCausalLM,TracConfig
from transformers import AutoModelForCausalLM
def load_model(config, pretrained_path=None, lora=False):
    base_model_name = config["config"]["language_model"]["name"]
    print("base_model_name", base_model_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    new_tokens = ["<image>", "<PAD>"]
    special_tokens_dict = {}
    if "<image>" not in tokenizer.get_vocab():
        special_tokens_dict["additional_special_tokens"] = ["<image>"]
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<PAD>"

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Tokenizer vocab size after adding: {len(tokenizer)}")
    print(f"pad_token = {tokenizer.pad_token}")

    custom_config = config["config"]

    if not pretrained_path:
        print("Loading TracLlavaForCausalLM from scratch")
        trac_config = TracConfig(custom_config)
        model = TracLlavaForCausalLM(trac_config, tokenizer=tokenizer)
        model.adjust_embeddings_from_num_new_tokens(len(tokenizer) - model.model.config.vocab_size)
        return tokenizer, model

    print("Loading pretrained model from:", pretrained_path)
    trac_config = TracConfig(custom_config)

    if lora:
        print("Loading LoRA weights")
        from peft import PeftModel

        base_model = TracLlavaForCausalLM(trac_config, tokenizer=tokenizer)
        base_model.adjust_embeddings_from_num_new_tokens(len(tokenizer) - base_model.model.config.vocab_size)
        model = PeftModel.from_pretrained(base_model, pretrained_path)
        return tokenizer, model

    else:
        print("Loading full model weights into wrapper")
        model = TracLlavaForCausalLM(trac_config, tokenizer=tokenizer)
        state_dict = torch.load(f"{pretrained_path}/pytorch_model.bin", map_location="cpu")

        model.model.resize_token_embeddings(len(tokenizer))

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

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
    pretrain_path="output/6xtl6uw0/checkpoint-483"
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
    
    