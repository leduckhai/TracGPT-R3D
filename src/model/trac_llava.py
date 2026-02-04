import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
import sys
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM,AutoModel
import os
from dotenv import load_dotenv
load_dotenv()
ROOT=os.getenv("ROOT")
sys.path.append(ROOT)
from transformers.generation.utils import GenerationMixin
from src.model.vision_encoder.load_encoder import load_vision_encoder
from src.model.projector.projector import load_mm_projector
import numpy as np
from src.model.base_model import BaseModel
import torch.nn.functional as F
from transformers import LlamaConfig

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

IGNORE_INDEX = -100
IMAGE_TOKEN_ID = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

class TracConfig(LlamaConfig):
    model_type = "trac"

    def __init__(self, config=None, **kwargs):
        self.custom_config = config if config is not None else {}
        super().__init__(**kwargs)

class TracLlavaForCausalLM(GenerationMixin, PreTrainedModel,BaseModel):
    config_class = None 
    def __init__(self, config, tokenizer=None):
        BaseModel.__init__(self, tokenizer)
        super().__init__(config)
        torch.manual_seed(42)

        cfg = config.custom_config
        base_model_name = cfg["language_model"]["name"]
        self.tokenizer = tokenizer
        self.alpha = 0.5  

        self.lm_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
        )
        self.embed_tokens=self.lm_model.model.embed_tokens

        old_emb = self.lm_model.get_input_embeddings()
        old_num, hidden_size = old_emb.weight.shape
        new_num = len(tokenizer)

        if new_num != old_num:
            print("new num not equal to old num, resizing embeddings", new_num, old_num)
            self.lm_model.resize_token_embeddings(new_num)

        self.vision_encoder = load_vision_encoder(cfg["vision_encoder"])
        self.mm_projector = load_mm_projector(cfg["projector"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.vision_encoder.to(device)
        self.mm_projector.to(device)

        self.freeze_vision_encoder = True
        if self.freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        self.n_class = 3

    def freeze_llm(self):
        for name, p in self.lm_model.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = False

    def unfreeze_llm(self):
        for name, p in self.lm_model.named_parameters():
            if "lora" in name.lower():
                p.requires_grad = True
        
    def adjust_embeddings_from_num_new_tokens(self, num_new_tokens: int):
        with torch.no_grad():
            embeddings = self.lm_model.get_input_embeddings().weight
            if num_new_tokens > 0:
                old_embeddings_avg = embeddings[:-num_new_tokens, :].mean(dim=0)
                embeddings[-num_new_tokens:, :] = old_embeddings_avg
                
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        images=None,
        attention_mask=None,
        position_ids=None,
        max_new_tokens=128,
        num_beams=1,
        do_sample=False,
        **kwargs
    ):
        inputs_embeds, _, attention_mask, position_ids = self.prepare_input(
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        position_ids=position_ids,
        )

        inputs_embeds = inputs_embeds.to(self._device)
        attention_mask = attention_mask.to(self._device)
        if position_ids is not None:
            position_ids = position_ids.to(self._device)

        return self.lm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs
        )
       
    def get_image_embeddings(self, images):
        return self.encode_single_image(images)
    
    def classify(self,images):
        return self.classifier(images)
        
    def forward(self, input_ids=None,attention_mask=None, labels=None, **kwargs):
        device = next(self.parameters()).device
        images=kwargs.get("images", None)
        # image_features=kwargs.get("image_features", None)
        if images is not None:
            inputs_embeds, labels, attention_mask, position_ids = self.prepare_input(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                labels=labels,
                # image_features=image_features
                
            )
            inputs_embeds = inputs_embeds.to(device)
            input_ids = None  
        else:
            inputs_embeds = kwargs.get("inputs_embeds", None)
            # if input_ids is not None:
            #     input_ids = input_ids.to(device)
        outputs = self.lm_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            #  output_hidden_states=True,
             return_dict=True,
        )
        return outputs
        
def decode_labels(tokenizer, labels_tensor):
    labels = labels_tensor.clone()
    labels[labels == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(labels, skip_special_tokens=True)
        

if __name__ == "__main__":
    import sys

    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    import yaml

    # config_path="/root/repo/TracGPT-R3D/config/vit_llama_3B.yaml"
    config_path="/root/repo/TracGPT-R3D/config/vit_llama_3B_cot_lite.yaml"
    # config_path = "/workspace/TracGPT-R3D/config/vit_llama_3B_80GB.yaml"
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    custom_config = full_config["model"]["config"]
    base_model_name = custom_config["language_model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model_name = custom_config["language_model"]["name"]
    print("Loading base model:", base_model_name)

    new_tokens = ["<image>", "<PAD>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    tokenizer.pad_token = "<PAD>"
    print("pad token id:", tokenizer.pad_token_id)

    config = TracConfig(custom_config)
    model = TracLlavaForCausalLM(config, tokenizer=tokenizer)
    model.adjust_embeddings_from_num_new_tokens(len(new_tokens))
    model.to("cuda")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nAfter modification:")
    print("pad_token:", tokenizer.pad_token)
    print("padding_side:", tokenizer.padding_side)
    print("vocab_size:", len(tokenizer))
    print("eos_token_id:", tokenizer.eos_token_id)
    data_config = full_config["data"]
    train_set, val_set, test_set = load_data(
        train_val_dir=data_config["train_val_dir"],
        test_dir=data_config["test_dir"],
        image_train_path=data_config["image_train_path"],
        image_test_path=data_config["image_test_path"],
        dataset=data_config["dataset"],
        train_sample=data_config["train_sample"],
        val_sample=data_config["val_sample"],
        test_sample=data_config["test_sample"],
    )

    collator = load_collator(full_config["general"]["collator"], tokenizer=tokenizer)
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
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    for i,batch in enumerate(train_loader):
        with torch.no_grad():
            print("train loader mode")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)
            text = batch["full_texts"]
            answer = batch["class_labels"]
            aux_labels = batch["aux_labels"].to(device)
            print("aux_labels", aux_labels)
            decode_input=tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            decode_label = decode_labels(tokenizer, labels)
            print("decoded input_ids", decode_input)
            print("decoded labels", decode_label)
            # image_features = model.get_image_embeddings(images)
            # output=model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     labels=labels,
            #     # image_features=image_features,
            #     images=images,
            #     aux_labels=aux_labels
                
            # )
            # # print("output", output)
            # # logit=model.classify(image_features)
            # # print("logit", logit)
            # # print("output loss", output.loss)
            # gen_output=model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     images=images,
            #     max_new_tokens=50,
            #     pad_token_id=tokenizer.pad_token_id,
            #     eos_token_id=tokenizer.eos_token_id
            # )
            # gen_text=tokenizer.batch_decode(gen_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # print("gen_text", gen_text)
            if i==4:
                break
    for i,batch in enumerate(val_loader):
        with torch.no_grad():
            print("val loader mode")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)
            text = batch["full_texts"]
            answer = batch["class_labels"]
            decode_input=tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            decode_label = decode_labels(tokenizer, labels)
            print("decoded input_ids", decode_input)
            print("decoded labels", decode_label)
            # print("input_ids", input_ids, "labels", labels, "attention_mask", attention_mask)
            
            # output=model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     labels=labels,
            #     images=images
            # )
              
            # print("output loss", output.loss)
            if i==4:
                break
    # for i,batch in enumerate(test_loader):  
    #     with torch.no_grad():
    #         print("test loader mode")
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         labels = batch["labels"].to(device)
    #         images = batch["images"].to(device)
    #         text = batch["full_texts"]
    #         answer = batch["class_labels"]
    #         decode_input=tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    #         decode_label = decode_labels(tokenizer, labels)
    #         print("decoded input_ids", decode_input)
    #         print("decoded labels", decode_label)
    #         # print("input_ids", input_ids, "labels", labels, "attention_mask", attention_mask)
            
    #         output=model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             labels=labels,
    #             images=images
    #         )
    #         if i==4:
    #             break
        
    # for i,batch in enumerate(val_loader):
    #     print("val loader mode")
    #     with torch.no_grad():
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         images = batch["images"].to(device)
    #         text = batch["full_texts"]
    #         answer = batch["class_labels"]
    #         print("full text",text)
    #         # print("non neg indices", non_neg_indices)
    #         # print("corresponding input_ids", input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]])
    #         # print("decoded corresponding input_ids", tokenizer.batch_decode(input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]], skip_special_tokens=False))
    #         # print("input_ids", input_ids, "labels", labels, "attention_mask", attention_mask)
    #         print("input_id", input_ids.shape, input_ids)
    #         print("attention_mask", attention_mask.shape, attention_mask)
    #         print("images", images.shape)
    #         output=model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             images=images
    #         )
    #         if i==10:
    #             break
    # for i,batch in enumerate(test_loader):
    #     print("test loader mode")
    #     with torch.no_grad():
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
            
    #         labels = batch["labels"].to(device)
    #         images = batch["images"].to(device)
    #         text = batch["full_texts"]
    #         answer = batch["class_labels"]
    #         print("input",text)
    #         raw_text=tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    #         print("raw_text", raw_text)
    #         if i==6:
    #             break
        
        # print("input_ids", input_ids)
        # class_labels = batch["class_labels"]
        # # output=model(
        # #     input_ids=input_ids,
        # #     attention_mask=attention_mask,
        # #     labels=labels,
        # #     # images=images
        # # )
        # raw_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # print("raw_text", raw_text)
        # break
    # for batch in val_loader:
    #     print("val loader mode")
    #     input_ids = batch["input_ids"].to(device)
    #     attention_mask = batch["attention_mask"].to(device)
    #     labels = batch["labels"].to(device)
    #     images = batch["images"].to(device)

    #     text = batch["full_texts"]
    #     answer = batch["class_labels"]
    #     print("answer", answer)
    #     # raw_text=tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    #     # print("raw_text", raw_text)
    #     # print("input_ids", input_ids)
    #     # output=model(
    #     #     input_ids=input_ids,
    #     #     attention_mask=attention_mask,
    #     #     labels=labels,
    #     #     # images=images
    #     # )
    #     break
    # answer_set=set()
    # for batch in test_loader:
    #     print("test loader mode")
    #     input_ids = batch["input_ids"].to(device)
    #     print(
    #         "decoded input_ids",
    #         tokenizer.batch_decode(input_ids, skip_special_tokens=True),
    #     )
    #     attention_mask = batch["attention_mask"].to(device)
    #     labels = batch["labels"].to(device)
    #     images = batch["images"].to(device)

    #     text = batch["full_texts"]
    #     answer = batch["class_labels"]
    #     for a in answer:
    #         answer_set.add(a)
        # raw_text=tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # print("input_ids", input_ids)
        # print("raw_text", raw_text)
        # print
        # print("text", text)
        # answer=batch["class_labels"]

        # with torch.inference_mode():
        #     prompt = "Question: What is the symtompt? "
        #     inputs = tokenizer(prompt, return_tensors="pt")

        #     # Generate
        #     input_ids=inputs.input_ids
        #     attention_mask=inputs.attention_mask
        #     input_ids=input_ids.to("cuda")
        #     attention_mask=attention_mask.to("cuda")

        #     # generate_ids = model.generate(
        #     #     input_ids=input_ids,
        #     #     attention_mask=attention_mask,
        #     #     max_new_tokens=50,
        #     #     pad_token_id=tokenizer.pad_token_id,)

        #     generate_ids = model.generate(
        #         input_ids=input_ids,
        #         images=images,
        #         attention_mask=attention_mask,
        #         max_new_tokens=50,
        #         pad_token_id=tokenizer.pad_token_id,)

        #     output=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #     print("output", output)

        # break
    # print("answer_set", answer_set)