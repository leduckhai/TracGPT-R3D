import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
import sys
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import os
from dotenv import load_dotenv
load_dotenv()
ROOT=os.getenv("ROOT")
sys.path.append(ROOT)
from transformers.generation.utils import GenerationMixin
from src.model.vision_encoder.load_encoder import load_vision_encoder
from src.model.projector.projector import load_mm_projector
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LlamaForCausalLM
from transformers import LlamaConfig
import numpy as np
from src.model.base_model import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from src.model.arch import TracMetaModel, TracMetaForCausalLM
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_ID = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"


class TracConfig(LlamaConfig):
    model_type = "trac_llama"


class TracLlamaModel(LlamaModel,TracMetaModel):
    config_class = TracConfig
    def __init__(self, config: LlamaConfig):
        super(TracLlamaModel, self).__init__(config)

class TracLLamaForCausalLM(TracMetaForCausalLM, LlamaForCausalLM):
    config_class = TracConfig
    def __init__(self, config):
        super(TracLLamaForCausalLM, self).__init__(config)
        self.model=TracLlamaModel(config)
        self.vocab_size = config.vocab_size
    def get_model(self):
        return self.model
    def forward(
        self,
        input_ids,
        images=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):

        if inputs_embeds is None:
            inputs_embeds, labels, attention_mask, position_ids = (
                self.prepare_input(
                    input_ids=input_ids,
                    images=images,
                    labels=labels,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                )
            )

        transformer_outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
                    
        return transformer_outputs
    
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
        """
        Custom generate wrapper that preprocesses multimodal input
        and calls HF's generate.
        """
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        if images is not None:
            inputs_embeds, labels, attention_mask, position_ids = (
                self.prepare_input(
                    input_ids=input_ids,
                    images=images,
                    labels=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                )
            )

            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
       
                do_sample=do_sample,
               
                **kwargs
            )
        else:
            outputs = super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
               
                **kwargs
            )

        return outputs

        
AutoConfig.register("lamed_llama", TracConfig)
AutoModelForCausalLM.register(TracConfig, TracLLamaForCausalLM)
if __name__ == "__main__":
    import sys

    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    import yaml

    config_path="/root/TracGPT-R3D/config/vit_llama_3B.yaml"
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
        dataset_config=data_config["dataset_config"],
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

    answer_set=set()

    file_path="debug_dump.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted.")
    else:
        print(f"{file_path} does not exist.")
    print("projector", model.mm_projector)
    for i,batch in enumerate(train_loader):
        with torch.no_grad():
            print("train loader mode")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)
            text = batch["full_texts"]
            answer = batch["class_labels"]
            print("full text",text)
            non_neg_indices = torch.nonzero(labels != -100, as_tuple=False)
            print("corresponding input_ids", input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]])
            print("decoded corresponding input_ids", tokenizer.batch_decode(input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]], skip_special_tokens=False))
            # print("input_ids", input_ids, "labels", labels, "attention_mask", attention_mask)
            
            output=model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images
            )
            print("output", output)
            generate_ids = model.generate(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,)

            output=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print("output", output)
            break
           
    # for i,batch in enumerate(train_loader):
    #     with torch.no_grad():
    #         print("train loader mode")
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         labels = batch["labels"].to(device)
    #         images = batch["images"].to(device)
    #         text = batch["full_texts"]
    #         answer = batch["class_labels"]
    #         print("full text",text)
    #         non_neg_indices = torch.nonzero(labels != -100, as_tuple=False)
    #         print("corresponding input_ids", input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]])
    #         print("decoded corresponding input_ids", tokenizer.batch_decode(input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]], skip_special_tokens=False))
    #         # print("input_ids", input_ids, "labels", labels, "attention_mask", attention_mask)
            
    #         # output=model(
    #         #     input_ids=input_ids,
    #         #     attention_mask=attention_mask,
    #         #     labels=labels,
    #         #     # images=images
    #         # )
    #         if i==20:
    #             break
    # for i,batch in enumerate(val_loader):
    #     with torch.no_grad():
    #         print("val loader mode")
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         labels = batch["labels"].to(device)
    #         images = batch["images"].to(device)
    #         text = batch["full_texts"]
    #         answer = batch["class_labels"]
    #         print("full text",text)
    #         non_neg_indices = torch.nonzero(labels != -100, as_tuple=False)
    #         print("corresponding input_ids", input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]])
    #         print("decoded corresponding input_ids", tokenizer.batch_decode(input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]], skip_special_tokens=False))
    #         # print("input_ids", input_ids, "labels", labels, "attention_mask", attention_mask)
            
    #         # output=model(
    #         #     input_ids=input_ids,
    #         #     attention_mask=attention_mask,
    #         #     labels=labels,
    #         #     images=images
    #         # )
    #         if i==10:
    #             break
    # for i,batch in enumerate(test_loader):  
    #     with torch.no_grad():
    #         print("test loader mode")
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         labels = batch["labels"].to(device)
    #         images = batch["images"].to(device)
    #         text = batch["full_texts"]
    #         answer = batch["class_labels"]
    #         print("full text",text)
    #         non_neg_indices = torch.nonzero(labels != -100, as_tuple=False)
    #         print("corresponding input_ids", input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]])
    #         print("decoded corresponding input_ids", tokenizer.batch_decode(input_ids[non_neg_indices[:, 0], non_neg_indices[:, 1]], skip_special_tokens=False))
    #         # print("input_ids", input_ids, "labels", labels, "attention_mask", attention_mask)
            
    #         # output=model(
    #         #     input_ids=input_ids,
    #         #     attention_mask=attention_mask,
    #         #     labels=labels,
    #         #     images=images
    #         # )
    #         if i==10:
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