from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import sys
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

sys.path.append("/root/TracGPT-R3D")
from src.model.projector.projector import load_mm_projector
from src.model.vision_encoder.load_encoder import load_vision_encoder
from typing import Optional


class TracLlamaConfig:
    model_type = "trac-llama"

    def __init__(self, config: dict):
        self.config = config

    # model_name = "meta-llama/Llama-3.2-1B"

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
    

class TracLlamaForCausalLM(PreTrainedModel):
    config_class = TracLlamaConfig

    def __init__(self, config):
        super().__init__(config)

        self.vision_encoder = load_vision_encoder(config["vision_encoder"])
        self.mm_projector = load_mm_projector(config["projector"])

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Process vision input
        # vision_features = self.vit(images).last_hidden_state[:, 0, :]  # CLS token
        # vision_features = self.vision_proj(vision_features)
        vision_features = self.vision_encoder(images)
        vision_features = self.mm_projector(vision_features)
        text_embeddings = 
        # multi_model_input=
        # Process text input
        text_features = self.llama(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        text_features = text_features[:, -1, :]  # Last token
        text_features = self.text_proj(text_features)

        # Fuse modalities
        fused = torch.cat([vision_features, text_features], dim=-1)
        fused = self.fusion(fused)

        # For generation tasks, you might return fused features
        # For classification, add:
        # logits = self.classifier(fused)
        # return logits

        return fused
