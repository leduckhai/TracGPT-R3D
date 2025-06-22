from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from dotenv import load_dotenv
import os 
# Load from .env file
load_dotenv()
ROOT=os.getenv("ROOT")
import sys
sys.path.append(ROOT)
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, Phi3Config, Phi3Model,PhiModel,PhiConfig,PhiForCausalLM,Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import AutoConfig, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Union, Any, Dict
import json
from model.multimodel_processor import MultimodalProcessor
from model.Trac_phi_config import TracPhi3Config, MultimodalConfig


class VisionEncoder(nn.Module):
    """Handles vision encoding and projection"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.vision_tower = None
        self.mm_projector = None
        self._is_built = False
    
    def build_components(self, model_args=None):
        """Build vision tower and projector"""
        print("Building multimodal vision tower, mm_projector components...")
        if self._is_built:
            return
        if self.config.vision_tower:
            try:
                self.vision_tower = self._build_vision_tower()
                print("Vision tower built successfully.")
                self.mm_projector = self._build_mm_projector()
                print("MM projector built successfully.")
                self._is_built = True
            except Exception as e:
                print(f"Warning: Failed to build multimodal components: {e}")
    
    def _build_vision_tower(self):
        """Build vision tower - implement based on your builder"""
        try:
            from model.Encoder.encoder import build_vision_tower
            return build_vision_tower(self.config)
        except ImportError as e:
            raise ImportError(f"Failed to import vision tower builder: {e}")
    
    def _build_mm_projector(self):
        """Build multimodal projector - implement based on your builder"""
        try:
            from model.Projector.projector import build_mm_projector
            return build_mm_projector(self.config)
        except ImportError as e:
            raise ImportError(f"Failed to import mm projector builder: {e}")
    
    def encode_images(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode images to features"""
        if not self._is_built or images is None:
            return None
        
        try:
            image_features = self.vision_tower(images)
            print("image_features 1", image_features.shape)
            image_features = self.mm_projector(image_features)
            print("image_features 2", image_features.shape)
            return image_features
        except Exception as e:
            print(f"Warning: Failed to encode images: {e}")
            return None
    
    def load_pretrained_weights(self, vision_path: str, projector_path: str):
        """Load pretrained weights"""
        if vision_path and self.vision_tower:
            try:
                weights = torch.load(vision_path, map_location='cpu')
                self.vision_tower.load_state_dict(weights, strict=True)
                print(f"Loaded vision weights from {vision_path}")
            except Exception as e:
                print(f"Warning: Failed to load vision weights from {vision_path}: {e}")
        
        if projector_path and self.mm_projector:
            try:
                weights = torch.load(projector_path, map_location='cpu')
                # Extract projector weights
                projector_weights = {
                    k.split('mm_projector.')[1]: v 
                    for k, v in weights.items() 
                    if 'mm_projector' in k
                }
                if projector_weights:
                    self.mm_projector.load_state_dict(projector_weights, strict=True)
                    print(f"Loaded projector weights from {projector_path}")
                else:
                    print(f"Warning: No projector weights found in {projector_path}")
            except Exception as e:
                print(f"Warning: Failed to load projector weights from {projector_path}: {e}")


class BBox3DPredictor(nn.Module):
    """Handles 3D bounding box prediction"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.bbox3d_head = None
        self.bbox3d_projector = None
        self.enabled = False
        self.loss_calculator = None
        
        if config.bbox3d_module:
            try:
                self._build_components()
            except Exception as e:
                print(f"Warning: Failed to build bbox3d components: {e}")
    
    def _build_components(self):
        """Build bbox3d components"""
        self.bbox3d_head = self._build_bbox3d_head()
        self.bbox3d_projector = nn.Sequential(
            nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mm_hidden_size, self.config.mm_hidden_size),
            nn.Dropout(0.1),
        )
        self.loss_calculator = self._create_loss_calculator()
        self.enabled = True
    
    def _build_bbox3d_head(self):
        """Build bbox3d head - implement based on your builder"""
        try:
            from model.bbox3d.builder import build_bbox3d_module
            return build_bbox3d_module(self.config)
        except ImportError as e:
            raise ImportError(f"Failed to import bbox3d builder: {e}")
    
    def _create_loss_calculator(self):
        """Create loss calculation module"""
        try:
            from model.loss import L1Loss, IoU3DLoss
            
            class BBox3DLossCalculator(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.l1_loss = L1Loss()
                    self.iou3d_loss = IoU3DLoss()
                
                def compute_loss(self, predictions, targets, masks=None):
                    """Compute bbox3d loss with proper masking"""
                    if masks is not None:
                        # Apply mask to compute loss only on valid boxes
                        valid_mask = masks.unsqueeze(-1).float()
                        masked_pred = predictions * valid_mask
                        masked_target = targets * valid_mask
                        
                        l1_loss = self.l1_loss(masked_pred, masked_target)
                        iou_loss = self.iou3d_loss(masked_pred, masked_target)
                        
                        # Normalize by number of valid boxes
                        num_valid = masks.sum().clamp(min=1)
                        return (l1_loss + iou_loss) / num_valid
                    else:
                        return self.l1_loss(predictions, targets) + self.iou3d_loss(predictions, targets)
            
            return BBox3DLossCalculator()
        except ImportError:
            # Fallback loss calculator
            return nn.MSELoss()
    
    def extract_bbox_features(self, hidden_states: torch.Tensor, 
                            bbox_token_mask: torch.Tensor) -> torch.Tensor:
        """Extract bbox features from hidden states"""
        bbox_prompts = []
        
        for i in range(bbox_token_mask.shape[0]):
            token_count = torch.sum(bbox_token_mask[i])
            
            if token_count == 1:
                bbox_token = hidden_states[i][bbox_token_mask[i]]
                bbox_prompt = self.bbox3d_projector(bbox_token)
            elif token_count > 1:
                bbox_tokens = hidden_states[i][bbox_token_mask[i]]
                bbox_token = torch.mean(bbox_tokens, dim=0, keepdim=True)
                bbox_prompt = self.bbox3d_projector(bbox_token)
            else:
                bbox_prompt = torch.zeros(
                    [1, self.config.mm_hidden_size],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )
            bbox_prompts.append(bbox_prompt)
        
        return torch.cat(bbox_prompts, dim=0)
    
    def predict_bboxes(self, vision_features: torch.Tensor, 
                      text_features: torch.Tensor) -> torch.Tensor:
        """Predict 3D bounding boxes"""
        if not self.enabled:
            return None
        
        try:
            # Pool features
            vision_pooled = vision_features.mean(dim=1)  # [B, D_v]
            text_pooled = text_features.mean(dim=1)      # [B, D_t]
            
            # Combine features
            combined = torch.cat([vision_pooled, text_pooled], dim=-1)
            
            # Predict bboxes
            return self.bbox3d_head(combined)
        except Exception as e:
            print(f"Warning: Failed to predict bboxes: {e}")
            return None
    
    def compute_bbox_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                         masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute bbox prediction loss with proper shape handling"""
        if not self.enabled or predictions is None:
            return torch.tensor(0.0, device=predictions.device if predictions is not None else 'cpu')
        
        try:
            # Handle different prediction shapes
            if predictions.dim() == 2 and targets.dim() == 3:
                # Pred: [B, 6], Target: [B, max_boxes, 6]
                if masks is not None:
                    # Use first valid box as target
                    valid_indices = masks.argmax(dim=1)  # Get first valid box index
                    targets = targets[torch.arange(targets.size(0)), valid_indices]
                else:
                    targets = targets[:, 0, :]  # Use first box
            
            elif predictions.dim() == 3 and targets.dim() == 3:
                # Both: [B, max_boxes, 6]
                if masks is not None:
                    loss_full = F.smooth_l1_loss(predictions, targets, reduction='none')
                    loss_masked = loss_full * masks.unsqueeze(-1).float()
                    return loss_masked.sum() / masks.sum().clamp(min=1)
            
            return self.loss_calculator.compute_loss(predictions, targets, masks)
        except Exception as e:
            print(f"Warning: Failed to compute bbox loss: {e}")
            return torch.tensor(0.0, device=predictions.device)

class TracPhi3Model(Phi3Model):
    """Trac Phi3 base model with multimodal capabilities"""
    
    config_class = TracPhi3Config
    
    def __init__(self, config: TracPhi3Config):
        super().__init__(config)
        self.config = config
        self.multimodal_config = config.multimodal
        print("multimodal_config", self.multimodal_config)        
        # Initialize multimodal components
        self.vision_encoder = VisionEncoder(self.multimodal_config)
        self.bbox3d_predictor = BBox3DPredictor(self.multimodal_config)
        self.multimodal_processor = MultimodalProcessor(self.vision_encoder)
        self.multimodal_processor.set_image_tokens(
          
            self.multimodal_config.img_token_id, 
        )
    
    def initialize_multimodal_components(self, model_args):
        """Initialize all multimodal components"""
        self._update_config_from_args(model_args)
        self.vision_encoder.build_components(model_args)
        
        # Load pretrained weights if provided
        if hasattr(model_args, 'pretrain_vision_model') and model_args.pretrain_vision_model:
            vision_path = model_args.pretrain_vision_model
            projector_path = getattr(model_args, 'pretrain_mm_mlp_adapter', None)
            self.vision_encoder.load_pretrained_weights(vision_path, projector_path)
    
    def _update_config_from_args(self, model_args):
        """Update config from model arguments"""
        config_mappings = {
            'image_channel': 'image_channel',
            'image_size': 'image_size', 
            'patch_size': 'patch_size',
            'vision_tower': 'vision_tower',
            'vision_select_layer': 'vision_select_layer',
            'vision_select_feature': 'vision_select_feature',
            'mm_projector_type': 'mm_projector_type',
            'proj_layer_type': 'proj_layer_type',
            'proj_layer_num': 'proj_layer_num',
            'proj_pooling_type': 'proj_pooling_type',
            'proj_pooling_size': 'proj_pooling_size'
        }
        
        for attr, config_key in config_mappings.items():
            if hasattr(model_args, attr):
                setattr(self.multimodal_config, config_key, getattr(model_args, attr))


class TracPhi3ForCausalLM(Phi3ForCausalLM):
    """Trac Phi3 for causal language modeling with 3D bbox prediction"""
    
    config_class = TracPhi3Config
    
    def __init__(self, config: TracPhi3Config):
        super().__init__(config)
        self.model = TracPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def get_model(self):
        return self.model
    
    def all_to_device(self, device="cuda"):
        """Move all components to device"""
        self.model.to(device)
        if self.model.vision_encoder.vision_tower:
            self.model.vision_encoder.vision_tower.to(device)
        if self.model.vision_encoder.mm_projector:
            self.model.vision_encoder.mm_projector.to(device)
        if self.model.bbox3d_predictor.enabled:
            self.model.bbox3d_predictor.to(device)
    
    def prepare_inputs_for_multimodal(self, *args, **kwargs):
        """Delegate to multimodal processor"""
        return self.model.multimodal_processor.prepare_inputs_for_multimodal(
            *args, embed_tokens_fn=self.model.embed_tokens, **kwargs
        )
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        bbox_gts: Optional[torch.FloatTensor] = None,
        bbox_masks: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        answer_types: Optional[List] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        pre_input_ids = input_ids 
        
        (input_ids, position_ids, attention_masks, past_key_values, 
        inputs_embeds, labels, image_features) = self.prepare_inputs_for_multimodal(
            input_ids, images, position_ids=kwargs.get('position_ids'),
            attention_mask=attention_masks, past_key_values=kwargs.get('past_key_values'),
            labels=labels
        )
        
        outputs = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_masks,
            labels=labels,
            output_hidden_states=self.model.bbox3d_predictor.enabled,
            **{k: v for k, v in kwargs.items() 
               if k not in ['position_ids', 'past_key_values']}
        )
        
        if (self.model.bbox3d_predictor.enabled and 
            bbox_gts is not None and bbox_masks is not None):
            outputs = self._handle_bbox_prediction(
                outputs, image_features, bbox_gts, bbox_masks
            )
        
        return outputs
    
    def _handle_bbox_prediction(self, outputs, image_features, bbox_gts, bbox_masks):
        """Handle 3D bounding box prediction"""
        # Check for valid bbox samples
        if not bbox_masks.any():
            return outputs
        
        bbox_samples = bbox_masks.any(dim=1)
        if not bbox_samples.any():
            return outputs
        
        # Extract features for samples with bboxes
        vision_features = image_features[bbox_samples]
        text_features = outputs.hidden_states[-1][bbox_samples]
        targets = bbox_gts[bbox_samples]
        masks = bbox_masks[bbox_samples]
        
        # Predict bboxes
        bbox_predictions = self.model.bbox3d_predictor.predict_bboxes(
            vision_features, text_features
        )
        
        # Compute loss
        bbox_loss = self.model.bbox3d_predictor.compute_bbox_loss(
            bbox_predictions, targets, masks
        )
        
        # Add to outputs
        outputs.loss = outputs.loss + bbox_loss
        if hasattr(outputs, '__dict__'):
            outputs.bbox_3d_loss = bbox_loss
            outputs.bbox_3d_pred = bbox_predictions
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        bbox3d_enable: bool = False,
        **kwargs
    ):
        """Generate with optional 3D bbox prediction"""
        # Prepare inputs
        if images is not None:
            (inputs, _, attention_masks, _, inputs_embeds, _, 
             image_features) = self.prepare_inputs_for_multimodal(
                inputs, images, **kwargs
            )
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['attention_mask'] = attention_masks
        else:
            kwargs['inputs_embeds'] = self.model.embed_tokens(inputs)
        
        if bbox3d_enable and images is not None:
            return self._generate_with_bbox(images, **kwargs)
        else:
            return super().generate(**kwargs)
    
    def _generate_with_bbox(self, images, **kwargs):
        """Generate with 3D bbox prediction enabled"""
        outputs = super().generate(
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs
        )
        
        # Extract bbox features and predict
        bbox_logits = self._process_bbox_generation(outputs, images)
        
        return outputs.sequences, bbox_logits
    
    def _process_bbox_generation(self, outputs, images):
        """Process 3D bbox prediction during generation"""
        # Extract hidden states from generation
        last_tensors = [step[-1] for step in outputs.hidden_states]
        last_hidden_state = torch.cat(last_tensors[1:], dim=1)
        
        # Find bbox tokens
        bbox_token_mask = outputs.sequences[:, 1:] == self.config.multimodal.bbox3d_token_id
        
        # Extract bbox features
        bbox_prompts = self.model.bbox3d_predictor.extract_bbox_features(
            last_hidden_state, bbox_token_mask
        )
        
        # Predict bboxes
        logits = self.model.bbox3d_predictor.bbox3d_head(images, bbox_prompts)
        
        # Mask samples without bbox tokens
        no_bbox_ids = [i for i, mask in enumerate(bbox_token_mask) 
                      if not torch.sum(mask)]
        if no_bbox_ids:
            logits[no_bbox_ids] = -torch.inf
        
        return logits


# Registration
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("trac-phi3", TracPhi3Config)
AutoModelForCausalLM.register(TracPhi3Config, TracPhi3ForCausalLM)


