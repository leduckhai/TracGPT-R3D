import torch
from typing import Optional, List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import PreTrainedModel
import sys
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
sys.path.append("/root/TracGPT-R3D")
from transformers.generation.utils import GenerationMixin
from src.model.vision_encoder.load_encoder import load_vision_encoder
from src.model.projector.projector import load_mm_projector
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


class TracPhiConfig(PretrainedConfig):
    model_type = "trac-llama"

    def __init__(self, config=None, **kwargs):
        self.custom_config = config if config is not None else {}
        super().__init__(**kwargs)

class LlavaImageProcessor:
    def __init__(self, model, config, tokenizer=None):
        print("initializing image processor")
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device="cuda"
        self.image_token_name = "<image>"
        self.IGNORE_INDEX = -100
        self.IMAGE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(self.image_token_name)
        
        print("image token id", self.IMAGE_TOKEN_ID)
        
        self.vision_encoder = load_vision_encoder(config["vision_encoder"])
        self.mm_projector = load_mm_projector(config["projector"])
        
        self.vision_encoder.to(self.device)
        self.mm_projector.to(self.device)
        
        if config.get("freeze_vision_encoder", True):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def encode_single_image(self, image):
        """Encode a single image with safety checks"""
        image = image.to(self.device)
        
        
        # with torch.no_grad(): 
        image_features = self.vision_encoder(image)
        
    
        # Project features
        image_features = self.mm_projector(image_features)
        
        # Safety clamp to prevent extreme values
        image_features = torch.clamp(image_features, min=-10.0, max=10.0)
        image_features = torch.nan_to_num(image_features, nan=0.0)
        
        return image_features
    
    def embed_tokens(self, input_ids):
        return self.model.model.embed_tokens(input_ids)
    
    def prepare_input(self, input_ids, images, labels=None, attention_mask=None, 
                     position_ids=None, past_key_values=None):
        
        B, T = input_ids.shape
        
        image_features = self.encode_single_image(images)
        
        if image_features.dim() == 2:  # [B, D]
            image_features = image_features.unsqueeze(1)  # [B, 1, D]
            Vi = 1
        else:
            Vi = image_features.shape[1]  # [B, Vi, D]
        
        D = image_features.shape[2]
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if labels is None:
            labels = torch.full_like(input_ids, self.IGNORE_INDEX)

        # Embed text tokens -> [B, T, D]
        text_embeds = self.embed_tokens(input_ids)
        
        # Find image tokens
        image_token_id_tensor = torch.tensor(self.IMAGE_TOKEN_ID, 
                                           device=input_ids.device, 
                                           dtype=input_ids.dtype)
        is_image_token = (input_ids == image_token_id_tensor)  # [B, T]

        # Build output tensors
        num_image_tokens = is_image_token.sum(dim=1)  # [B]
        # print("num_image_tokens", num_image_tokens)
        # print("T", T,"Vi",Vi)
        expanded_lengths = (T - num_image_tokens) + num_image_tokens * Vi
        L = expanded_lengths.max().item()  # max expanded length

        input_embeds = torch.zeros((B, L, D), device=input_ids.device, dtype=text_embeds.dtype)
        new_labels = torch.full((B, L), self.IGNORE_INDEX, device=input_ids.device, dtype=labels.dtype)
        new_attention_mask = torch.zeros((B, L), device=input_ids.device, dtype=torch.bool)
        new_position_ids = torch.zeros((B, L), device=input_ids.device, dtype=torch.long)

        for b in range(B):
            seq_embeds = []
            seq_labels = []
            seq_masks = []
            pos_counter = 0

            for t in range(T):
                if is_image_token[b, t]:
                    # Use image features for this batch
                    seq_embeds.append(image_features[b])  # [Vi, D]
                    seq_labels.append(torch.full((Vi,), self.IGNORE_INDEX, 
                                               device=labels.device, dtype=labels.dtype))
                    seq_masks.append(torch.ones(Vi, dtype=torch.bool, device=input_ids.device))
                    pos_counter += Vi
                else:
                    if attention_mask[b, t]:  # Only include non-padded tokens
                        seq_embeds.append(text_embeds[b, t].unsqueeze(0))
                        seq_labels.append(labels[b, t].unsqueeze(0))
                        seq_masks.append(torch.ones(1, dtype=torch.bool, device=input_ids.device))
                        pos_counter += 1

            if seq_embeds:
                seq_embeds = torch.cat(seq_embeds, dim=0)
                seq_labels = torch.cat(seq_labels, dim=0)
                seq_masks = torch.cat(seq_masks, dim=0)
                
                current_length = seq_embeds.size(0)
                input_embeds[b, :current_length] = seq_embeds
                new_labels[b, :current_length] = seq_labels
                new_attention_mask[b, :current_length] = seq_masks
                new_position_ids[b, :current_length] = torch.arange(current_length, device=input_ids.device)

        # print("input_embeds shape", input_embeds, new_labels, new_attention_mask, new_position_ids)
        return input_embeds, new_labels, new_attention_mask, new_position_ids


class LlavaForCausalLM(GenerationMixin, PreTrainedModel):
    config_class = TracLlamaConfig
    
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        cfg = config.custom_config
        
        # Load model
        base_model_name=cfg["language_model"]["name"]
        print("Loading llama base model:", base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,  # Use half precision for stability
        )
        
        self.tokenizer = tokenizer
        if tokenizer and len(tokenizer) != self.model.config.vocab_size:
            self.model.model.resize_token_embeddings(len(tokenizer))
        
        self.image_processor = LlavaImageProcessor(self.model, cfg, tokenizer)
        
        # Add gradient clipping hook
        self.register_backward_hook(self._gradient_clipping_hook)
    
    def _gradient_clipping_hook(self, module, grad_input, grad_output):
        """Clip gradients to prevent explosion"""
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)
    
    def forward(self, input_ids, images=None, attention_mask=None, position_ids=None, 
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        
            
        
        if inputs_embeds is None:
            inputs_embeds, labels, attention_mask, position_ids = self.image_processor.prepare_input(
                input_ids=input_ids,  
                images=images,    
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values
            )
        
       
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
    
        return outputs

if __name__ == "__main__":
    import sys 
    

    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    import yaml
    
    config_path="/root/TracGPT-R3D/config/vit_llama_3B.yaml"
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)    
    custom_config = full_config["model"]["config"]
    base_model_name = custom_config["language_model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model_name = custom_config["language_model"]["name"]
    print("Loading base model:", base_model_name)
        
    new_tokens = ["<image>", "<PAD>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    tokenizer.pad_token = "<PAD>"
    config=TracLlamaConfig(custom_config)
    model = LlavaForCausalLM(config,tokenizer=tokenizer)
    model.to("cuda")
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nAfter modification:")
    print("pad_token:", tokenizer.pad_token)
    print("padding_side:", tokenizer.padding_side)

    
    data_config=full_config["data"]
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
    print("Collator:", full_config["general"]["collator"])
    collator = load_collator(full_config["general"]["collator"],tokenizer=tokenizer)
    train_loader=torch.utils.data.DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )
    test_loader=torch.utils.data.DataLoader(
        test_set,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )
    
    for batch in train_loader:
        print("train loader mode")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        images = batch["images"].to(device)
        
        text=batch["full_texts"]
        answer=batch["class_labels"]
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"   
        output=model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images
        )
    
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        images = batch["images"].to(device)
        
        text=batch["full_texts"]
        # print("text", text) 
        answer=batch["class_labels"]
        # print("input_ids", input_ids)
        # print("attention_mask", attention_mask)
        # print("labels", labels)
        # print("answer", answer)
        # print("images shape", images.shape)
        # print("Labels", labels, "input_ids", input_ids, "attention_mask", attention_mask)
      
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
        break
    
    
        