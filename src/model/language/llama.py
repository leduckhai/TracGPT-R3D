from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import sys
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import GenerationMixin
sys.path.append("/root/TracGPT-R3D")
from src.model.projector.projector import load_mm_projector
from src.model.vision_encoder.load_encoder import load_vision_encoder
from typing import Optional

from transformers.modeling_outputs import CausalLMOutputWithPast

class TracLlamaConfig(PretrainedConfig):
    model_type = "trac-llama"

    def __init__(self, config=None, **kwargs):
        self.custom_config = config if config is not None else {}
        super().__init__(**kwargs)

def prepare_multi_modal_input(image_embeddings, text_embeddings):
    """
    Prepare multi-modal input by concatenating image and text embeddings.
    
    Args:
        image_embeddings (torch.Tensor): Image embeddings of shape (batch_size, image_dim).
        text_embeddings (torch.Tensor): Text embeddings of shape (batch_size, text_dim).
        
    Returns:
        torch.Tensor: Concatenated embeddings of shape (batch_size, image_dim + text_dim).
    """
    return torch.cat((image_embeddings, text_embeddings), dim=-1)
    

from transformers.models.llama.modeling_llama import LlamaForCausalLM

class TracLlamaForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = TracLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        cfg = config.custom_config  

        self.vision_encoder = load_vision_encoder(cfg["vision_encoder"])
        self.mm_projector = load_mm_projector(cfg["projector"])
        self.language_model = LlamaForCausalLM.from_pretrained(cfg["language_model"]["name"])
        self.text_embed_fn = self.language_model.model.embed_tokens
        self.hidden_size = self.language_model.config.hidden_size
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
           
            batch_size = input_ids.size(0)

            vision_features = self.vision_encoder(images)        # (B, seq_v, dim_v)
            vision_features = self.mm_projector(vision_features) # (B, seq_v, hidden_size)

            text_embeddings = self.text_embed_fn(input_ids)      # (B, seq_t, hidden_size)

            multi_modal_input = torch.cat((vision_features, text_embeddings), dim=1) # (B, seq_v + seq_t, hidden)

            if attention_mask is not None:
                vision_mask = torch.ones(
                    batch_size, vision_features.size(1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat((vision_mask, attention_mask), dim=1)
            else:
                attention_mask = torch.ones(
                    batch_size, multi_modal_input.size(1),
                    dtype=torch.long,
                    device=multi_modal_input.device
                )

            # ---- 5. Handle positional embeddings
            # LLaMA uses rotary embeddings, which are applied inside the attention layers
            # To "shift" text positions, we just ensure inputs_embeds contains the correct sequence order
            # No manual pos_ids needed unless you want absolute embeddings
            position_ids = attention_mask.cumsum(dim=1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0) # Masked tokens keep pos=0
        else:
            multi_modal_input = inputs_embeds
        output = self.language_model(
            inputs_embeds=multi_modal_input,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        return output


    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, images=None, **kwargs):
        """
        This method is called automatically inside `.generate()` to get the right inputs for each decoding step.
        """
        batch_size = input_ids.size(0)

        # During generation, past_key_values are passed in kwargs
        past_key_values = kwargs.get("past_key_values", None)

        # Only encode vision features at the *first* decoding step
        if past_key_values is None and images is not None:
            vision_features = self.vision_encoder(images)
            vision_features = self.mm_projector(vision_features)

            text_embeddings = self.text_embed_fn(input_ids)
            inputs_embeds = torch.cat((vision_features, text_embeddings), dim=1)

            # Adjust attention mask
            if attention_mask is not None:
                vision_mask = torch.ones(batch_size, vision_features.size(1), device=attention_mask.device)
                attention_mask = torch.cat((vision_mask, attention_mask), dim=1)
        else:
            # During later steps, use only text token embeddings
            inputs_embeds = self.text_embed_fn(input_ids)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def generate_with_images(self, images, input_ids, attention_mask=None, **gen_kwargs):
        """
        Public helper for generation with multimodal input.
        """
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            **gen_kwargs
        )
if __name__ == "__main__":
    import yaml
    from transformers import AutoTokenizer
    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    config_path="/root/TracGPT-R3D/config/vit_llama.yaml"
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)    
    custom_config = full_config["model"]["config"]
    config = TracLlamaConfig(config=custom_config)
    
    base_model_name = custom_config["language_model"]["name"]
    print("Loading base model:", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = TracLlamaForCausalLM(config)
    device= "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for compatibility
    tokenizer.padding_side = "right"  # Ensure padding is on the right side
    batch_images = torch.randn(1, 1,32, 256, 256).to("cuda")
    batch_images = batch_images.to(device)
    prompt = "Describe this image in detail:"
    text_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # input_ids = text_inputs.input_ids
    # attention_mask = text_inputs.attention_mask
    
    data_config=full_config["data"]
    dataset= load_data(
        train_val_dir=data_config["train_val_dir"],
        test_dir=data_config["test_dir"],
        image_train_path=data_config["image_train_path"],
        image_test_path=data_config["image_test_path"],
        dataset=data_config["dataset"],
        train_sample=data_config["train_sample"],
        val_sample=data_config["val_sample"],
        test_sample=data_config["test_sample"],
        dataset_config=data_config["dataset_config"];
    )
    collator = load_collator(full_config["collator"])
    train_set, val_set, test_set = dataset
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    for i, batch in enumerate(train_loader):
        if i >= 1:
            break
        # images = batch["image"].to(device)
        # texts = batch["text"]
        # print("Batch images shape:", images.shape)
        # print("Batch text:", texts)
    outputs = model.generate_with_images(
        images=batch_images,
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
        max_length=50,
        temperature=0.7,
        top_p=0.9
    )

    # 5. Decode
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))