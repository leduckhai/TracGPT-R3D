from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()
ROOT = os.getenv("ROOT") or "/workspace/repo/TracGPT-R3D"
import sys

sys.path.append(ROOT)
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Phi3Model,
    Phi3ForCausalLM,
)

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
from model.bbox3d.builder_v2 import BBox3DPredictor


class VisionEncoder(nn.Module):
    """Handles vision encoding and projection"""

    def __init__(self, vision_tower_config="vit3d"):
        super().__init__()
        self.vision_tower_config = vision_tower_config
        self.vision_tower = None
        self.mm_projector = None
        self._is_built = False

    def build_components(self):
        """Build vision tower and projector"""
        print("Building multimodal vision tower, mm_projector components...")
        if self._is_built:
            return
        if self.vision_tower_config:
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

            return build_vision_tower(
                vision_tower=self.vision_tower_config,
            )
        except ImportError as e:
            raise ImportError(f"Failed to import vision tower builder: {e}")

    def _build_mm_projector(self):
        """Build multimodal projector - implement based on your builder"""
        try:
            from model.Projector.projector import build_mm_projector

            return build_mm_projector()
        except ImportError as e:
            raise ImportError(f"Failed to import mm projector builder: {e}")

    def encode_images(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode images to features"""
        if not self._is_built or images is None:
            return None

        try:
            # before: [2, 1, 32, 256, 256])
            image_features = self.vision_tower(images)
            print("image_features 1", image_features.shape)
            # ([2, 2048, 3072])
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
                weights = torch.load(vision_path, map_location="cpu")
                self.vision_tower.load_state_dict(weights, strict=True)
                print(f"Loaded vision weights from {vision_path}")
            except Exception as e:
                print(f"Warning: Failed to load vision weights from {vision_path}: {e}")

        if projector_path and self.mm_projector:
            try:
                weights = torch.load(projector_path, map_location="cpu")
                # Extract projector weights
                projector_weights = {
                    k.split("mm_projector.")[1]: v
                    for k, v in weights.items()
                    if "mm_projector" in k
                }
                if projector_weights:
                    self.mm_projector.load_state_dict(projector_weights, strict=True)
                    print(f"Loaded projector weights from {projector_path}")
                else:
                    print(f"Warning: No projector weights found in {projector_path}")
            except Exception as e:
                print(
                    f"Warning: Failed to load projector weights from {projector_path}: {e}"
                )


class TracPhi3Model(Phi3Model):
    """Trac Phi3 base model with multimodal capabilities"""

    config_class = TracPhi3Config

    def __init__(self, config: TracPhi3Config):
        super().__init__(config)
        self.config = config
        self.multimodal_config = config.multimodal
        self.vision_encoder = VisionEncoder()
        self.bbox3d_predictor = BBox3DPredictor()
        self.multimodal_processor = MultimodalProcessor(self.vision_encoder)
        self.multimodal_processor.set_image_tokens(
            self.multimodal_config.img_token_id,
        )

    def initialize_multimodal_components(self):
        """Initialize all multimodal components"""
        self.vision_encoder.build_components()

        # Load pretrained weights if provided
        # if (
        #     hasattr(model_args, "pretrain_vision_model")
        #     and model_args.pretrain_vision_model
        # ):
        #     vision_path = model_args.pretrain_vision_model
        #     projector_path = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        #     self.vision_encoder.load_pretrained_weights(vision_path, projector_path)


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
        mode: Optional[str] = "normal",
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        pre_input_ids = input_ids

        if mode == "normal":

            (
                input_ids,
                position_ids,
                attention_masks,
                past_key_values,
                inputs_embeds,
                labels,
                image_features,
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                images,
                position_ids=kwargs.get("position_ids"),
                attention_mask=attention_masks,
                past_key_values=kwargs.get("past_key_values"),
                labels=labels,
            )
        if image_features is None:
            print("image feature is none")
            return
        outputs = super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_masks,
            labels=labels,
            output_hidden_states=self.model.bbox3d_predictor.enabled,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["position_ids", "past_key_values"]
            },
        )
        # print("outputs", outputs.shape)
        if (
            self.model.bbox3d_predictor.enabled
            and bbox_gts is not None
            and bbox_masks is not None
        ):
            outputs = self._handle_bbox_prediction(
                outputs, image_features, bbox_gts, bbox_masks
            )

        return outputs

    def _handle_bbox_prediction(self, outputs, image_features, bbox_gts, bbox_masks):
        """Handle 3D bounding box prediction"""
        if not bbox_masks.any():
            return outputs

        bbox_samples = bbox_masks.any(dim=1)
        if not bbox_samples.any():
            print("No valid bbox samples found.")
            return outputs
        print(
            "image_features",
            image_features.shape,
            "bbox_samples",
            bbox_samples.shape,
            bbox_samples,
        )
        vision_features = image_features[bbox_samples]
        text_features = outputs.hidden_states[-1][bbox_samples]
        targets = bbox_gts[bbox_samples]
        masks = bbox_masks[bbox_samples]

        # Predict bboxes
        print(
            "vision_features",
            vision_features.shape,
            "text_features",
            text_features.shape,
        )
        bbox_predictions = self.model.bbox3d_predictor.predict_bboxes(
            vision_features, text_features
        )

        # Compute loss
        print("target", targets.shape, "mask", masks.shape,targets,"mask",masks)
        bbox_loss = self.model.bbox3d_predictor.compute_bbox_loss(
            bbox_predictions, targets, masks
        )

        # Add to outputs
        outputs.loss = outputs.loss + bbox_loss
        if hasattr(outputs, "__dict__"):
            outputs.bbox_3d_loss = bbox_loss
            outputs.bbox_3d_pred = bbox_predictions

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        bbox3d_enable: bool = False,
        **kwargs,
    ):
        """Generate with optional 3D bbox prediction"""
        # Prepare inputs
        if images is not None:
            (inputs, _, attention_masks, _, inputs_embeds, _, image_features) = (
                self.prepare_inputs_for_multimodal(inputs, images, **kwargs)
            )
            kwargs["inputs_embeds"] = inputs_embeds
            kwargs["attention_mask"] = attention_masks
        else:
            kwargs["inputs_embeds"] = self.model.embed_tokens(inputs)

        if bbox3d_enable and images is not None:
            return self._generate_with_bbox(images, **kwargs)
        else:
            return super().generate(**kwargs)

    def _generate_with_bbox(self, images, **kwargs):
        """Generate with 3D bbox prediction enabled"""
        outputs = super().generate(
            output_hidden_states=True, return_dict_in_generate=True, **kwargs
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
        bbox_token_mask = (
            outputs.sequences[:, 1:] == self.config.multimodal.bbox3d_token_id
        )

        # Extract bbox features
        bbox_prompts = self.model.bbox3d_predictor.extract_bbox_features(
            last_hidden_state, bbox_token_mask
        )

        # Predict bboxes
        logits = self.model.bbox3d_predictor.bbox3d_head(images, bbox_prompts)

        # Mask samples without bbox tokens
        no_bbox_ids = [
            i for i, mask in enumerate(bbox_token_mask) if not torch.sum(mask)
        ]
        if no_bbox_ids:
            logits[no_bbox_ids] = -torch.inf

        return logits


# Registration
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("trac-phi3", TracPhi3Config)
AutoModelForCausalLM.register(TracPhi3Config, TracPhi3ForCausalLM)
if __name__ == "__main__":
    from collator import QA3DDataset, BboxAwareCollator
    from torch.utils.data import DataLoader

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM

    model_max_length = 512
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    special_tokens = [
        "<im_patch>",
        "<bx_start>", 
        "<bx_end>",
        "<image>",
        "<image_newline>"
    ]

    if hasattr(tokenizer, 'add_special_tokens'):
        num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        print(f"Added {num_added} new special tokens")
    else:
        print("Warning: Tokenizer doesn't support adding special tokens")
    
    tokenizer.add_tokens("[SEG]")
    collator = BboxAwareCollator(
        tokenizer=tokenizer,
        max_length=model_max_length,
        max_bbox_length=9,
    )

    ds = QA3DDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collator)
    img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    config = TracPhi3Config(
        img_token_id=tokenizer.convert_tokens_to_ids("<im_patch>"),
        bbox3d_token_id=tokenizer.convert_tokens_to_ids("<bx_start>"),
    )
    # model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    # model_args.vocab_size = len(tokenizer)
    model = TracPhi3ForCausalLM(config)
    model.get_model().initialize_multimodal_components()
    model = model.to("cuda")
    for batch in dl:
        (
            images,
            input_ids,
            attention_mask,
            labels,
            bbox_gt,
            bbox_mask,
            answer_types,
            position_ids,
        ) = batch.values()
        images = images.to("cuda")
        input_ids = input_ids.to("cuda")
        input_ids[:, 2:5] = img_token_id
        attention_mask = attention_mask.to("cuda")
        labels = labels.to("cuda")
        bbox_gt = bbox_gt.to("cuda")
        bbox_mask = bbox_mask.to("cuda")
        answer_types = answer_types
        position_ids = position_ids.to("cuda")

        outputs = model(
            input_ids=input_ids,
            images=images,
            bbox_gts=bbox_gt,
            bbox_masks=bbox_mask,
            labels=labels,
            attention_masks=attention_mask,
            answer_types=answer_types,
            position_ids=position_ids,
        )
        # print("outputs:", outputs)
        break

    for batch in dl:
        (
            images,
            input_ids,
            attention_mask,
            labels,
            bbox_gt,
            bbox_mask,
            answer_types,
            position_ids,
        ) = batch.values()
        images = images.to("cuda")
        input_ids = input_ids.to("cuda")
        input_ids[:, 2:5] = img_token_id
        attention_mask = attention_mask.to("cuda")
        labels = labels.to("cuda")
        bbox_gt = bbox_gt.to("cuda")
        bbox_mask = bbox_mask.to("cuda")
        answer_types = answer_types
        position_ids = position_ids.to("cuda")

        with torch.no_grad():
            pass
            break

        # outputs = model(
        #     input_ids=input_ids,
        #     images=images,
        #     bbox_gts=bbox_gt,
        #     bbox_masks=bbox_mask,
        #     labels=labels,
        #     attention_masks=attention_mask,
        #     answer_types=answer_types,
        #     position_ids=position_ids,
        # )
        # # print("outputs:", outputs)
        # break
