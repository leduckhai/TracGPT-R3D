import torch
from typing import Optional, List
import torch
import torch.nn as nn
import sys
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

class LlavaImageProcessor:
    def __init__(self, model, config, tokenizer=None):
        print("initializing image processor")
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device if model else None
        self.image_token_name="<image>"
        self.IGNORE_INDEX = -100
        self.IMAGE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(self.image_token_name)  # Example token index for image placeholder
        
        print("image token id", self.IMAGE_TOKEN_ID)
        self.DEFAULT_IMAGE_PATCH_TOKEN = "<image>"
        self.DEFAULT_IM_START_TOKEN = "<im_start>"
        self.DEFAULT_IM_END_TOKEN = "<im_end>"
        self.vision_encoder=load_vision_encoder(config["vision_encoder"])
        self.mm_projector=load_mm_projector(config["projector"])
        self.vision_encoder.to(self.device)
        self.mm_projector.to(self.device)
    def encode_images(self, images):
        return torch.randn(images.shape[0], 256, 1024)
    def mm_projector(self, image_features):
        return image_features
    def encode_single_image(self, image):
        """Encode a single image for processing"""
        image_features=self.vision_encoder(image)
        print("image feature shape after encoder", image_features.shape)
        image_features = self.mm_projector(image_features)
        print("after mm projector stats",  image_features.shape,image_features.min(), image_features.max(),image_features.mean(), image_features.std())
        return image_features 
    
    def embed_tokens(self, input_ids):
        return self.model.model.embed_tokens(input_ids)
    def prepare_single_image_input(self, input_ids, image, labels=None, attention_mask=None, position_ids=None,past_key_values=None):
        """
        Prepare inputs for single image processing
        Returns: input_embeds, labels, attention_mask, position_ids
        """
        image_features = self.encode_single_image(image)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            
        if labels is None:
            labels = torch.full_like(input_ids, self.IGNORE_INDEX)

        print("before atn msk",input_ids)
        print("atn msk",attention_mask)
        input_ids = input_ids[attention_mask]
        print("after atn msk",input_ids)
        labels = labels[attention_mask]
        image_token_id_tensor = torch.tensor(self.IMAGE_TOKEN_ID, 
                                   device=input_ids.device,
                                   dtype=input_ids.dtype)
        image_token_positions = torch.where(input_ids ==image_token_id_tensor)[0]
        print("logging image token positions", image_token_positions)
        if len(image_token_positions) == 0:
            print("No image token found in input_ids")
            text_embeds = self.embed_tokens(input_ids)
            return text_embeds, labels, attention_mask, position_ids

        split_points = [-1] + image_token_positions.tolist() + [len(input_ids)]
        text_segments = []
        label_segments = []
        
        for i in range(len(split_points) - 1):
            start = split_points[i] + 1
            end = split_points[i + 1]
            if start < end:
                text_segments.append(input_ids[start:end])
                label_segments.append(labels[start:end])
        print("text segments", text_segments)
        text_embeds = [self.embed_tokens(segment) for segment in text_segments]
        print("logging text embeds shape", text_embeds[0].shape)
        combined_embeds = []
        combined_labels = []
        
        for i, text_embed in enumerate(text_embeds):
            combined_embeds.append(text_embed)
            combined_labels.append(label_segments[i])
            
            if i < len(image_token_positions):
                combined_embeds.append(image_features)
                combined_labels.append(torch.full((image_features.shape[0],), self.IGNORE_INDEX, 
                                                device=labels.device, dtype=labels.dtype))

        input_embeds = torch.cat(combined_embeds, dim=0)
        new_labels = torch.cat(combined_labels, dim=0)
        
        new_attention_mask = torch.ones(input_embeds.shape[0], dtype=torch.bool, device=input_embeds.device)
        new_position_ids = torch.arange(0, input_embeds.shape[0], dtype=torch.long, device=input_embeds.device)

        return input_embeds, new_labels, new_attention_mask, new_position_ids

    def initialize_vision_tokens(self, model_args):
        """Initialize special vision tokens in the tokenizer"""
        if not self.tokenizer:
            return

        if model_args.mm_use_im_patch_token:
            self.tokenizer.add_tokens([self.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = self.tokenizer.add_tokens([self.DEFAULT_IM_START_TOKEN, self.DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))


class LlavaForCausalLM(GenerationMixin, torch.nn.Module):
    def __init__(self, model, config, tokenizer=None):
        super().__init__()
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = LlavaImageProcessor(model, config, tokenizer)
    
    def forward(self,
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
                **kwargs):
        print("ipnut shape", input_ids.shape)
        if inputs_embeds is None:
            inputs_embeds, labels, attention_mask, position_ids = self.image_processor.prepare_single_image_input(
                input_ids=input_ids,  
                image=images,    
                labels=labels,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids
            )
        
        print("labels shape", labels.shape, labels.min(), labels.max(),labels.mean(), labels.std())
        print("input shape", inputs_embeds.shape, inputs_embeds.min(), inputs_embeds.max(),inputs_embeds.mean(), inputs_embeds.std())
        print("attention shape", attention_mask.shape, attention_mask.min(), attention_mask.max(),attention_mask.mean(), attention_mask.std())
        print("position shape", position_ids.shape, position_ids.min(), position_ids.max(),position_ids.mean(), position_ids.std())
        return self.model(
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
    @torch.no_grad()
    def generate(self, input_ids=None, images=None, attention_mask=None, **kwargs):
        """
        Generate text based on input text and optional single image
        """
        if images is not None and images.shape[0] == 1:
            inputs_embeds, _, attention_mask, position_ids = self.image_processor.prepare_single_image_input(
                input_ids.squeeze(0),
                images.squeeze(0),
                attention_mask=attention_mask.squeeze(0) if attention_mask is not None else None
            )
            
            inputs_embeds = inputs_embeds.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0) if attention_mask is not None else None
            position_ids = position_ids.unsqueeze(0) if position_ids is not None else None
            
            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs
            )
        else:
            # Text-only generation
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        Prepare inputs for generation, handling images if present
        """
        images = kwargs.pop("images", None)
        inputs = self.model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, **kwargs
        )
        
        if images is not None:
            inputs['images'] = images
            
        return inputs



if __name__ == "__main__":
    # demonstrate_usage()
    import sys 
    from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM

    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    import yaml
    from src.collators.standard_collator import StandardCollator
    
    config_path="/root/TracGPT-R3D/config/vit_llama.yaml"
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)    
    custom_config = full_config["model"]["config"]
    
    
    base_model_name = custom_config["language_model"]["name"]
    print("Loading base model:", base_model_name)
    llm_model=AutoModelForCausalLM.from_pretrained(base_model_name)
    config=None 
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        new_tokens = ["<image>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    new_vocab_size = len(tokenizer)
    print(f"New vocab size: {new_vocab_size}")
    llm_model.resize_token_embeddings(new_vocab_size)
    
    tokenizer.padding_side = "right"
    llm_model.to("cuda")
    # Now initialize the model with the updated tokenizer
    model = LlavaForCausalLM(llm_model, custom_config, tokenizer)

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
    model_name="meta-llama/Llama-3.2-1B"
    train_loader=torch.utils.data.DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )
    model.to(device)
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        images = batch["images"].to(device)
        print("images shape", images.shape)
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        