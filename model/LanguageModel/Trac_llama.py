from typing import List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
from dotenv import load_dotenv
import os
import yaml

load_dotenv()
ROOT = os.getenv("ROOT") or "/workspace/repo/TracGPT-R3D"
import sys

sys.path.append(ROOT)

from utils.type import dict_to_namespace
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import AutoConfig, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Union, Any, Dict
from model.multimodel_processor import MultimodalProcessor
from model.bbox3d.builder import BBox3DPredictor
from model.Encoder.encoder import build_vision_tower
from model.Projector.projector import build_mm_projector


class VisionEncoder(nn.Module):
    """Handles vision encoding and projection"""

    def __init__(self, config):
        super().__init__()
        # vision_tower_config="vit3d", model_tag=None
        # self.model_tag=model_tag
        self.vision_tower_config = config.vision_tower_config
        self.mm_projector_config = config.projector
        self.vision_tower = None
        self.mm_projector = None
        self._is_built = False

    def build_components(self):
        """Build vision tower and projector"""
        print("Building multimodal vision tower, mm_projector components...")
        if self._is_built:
            return
        if self.vision_tower_config:
            self.vision_tower = build_vision_tower(
                self.vision_tower_config,
            )
            print("Vision tower built successfully.")
            self.mm_projector = build_mm_projector(self.mm_projector_config)
            print("MM projector built successfully.")
            self._is_built = True
            
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


class TracLlama3Model(nn.Module):
    """Trac Phi3 base model with multimodal capabilities"""

    # config_class = TracPhi3Config

    def __init__(
        self, config, model_tag="tiny-llama", module_config_path="config/llama.yaml"
    ):
        super().__init__()
        with open(module_config_path, "r") as f:
            module_config = yaml.safe_load(f)
        module_config = dict_to_namespace(module_config)
        if model_tag == "tiny-llama":
            self.model_tag = "tiny-llama"
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.float32
            )
            self.model.resize_token_embeddings(config.vocab_size)
            self.module_configs = module_config.tiny_llama
        else:
            raise NotImplementedError

        self.vision_encoder = VisionEncoder(self.module_configs.vision_encoder)
        self.bbox3d_predictor = BBox3DPredictor(self.module_configs.bbox_predictor)
        self.multimodal_processor = MultimodalProcessor(
            self.vision_encoder, image_token_id=config.img_token_id
        )
        self.embed_tokens = self.model.model.embed_tokens

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

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


class TracLlamaForCausalLM(nn.Module):
    """Trac Phi3 for causal language modeling with 3D bbox prediction"""

    _model_cache = {}

    def __init__(self, config):
        print("init Trac llama")
        super().__init__()
        self.model = self.init_model(config)
        print("finish init base model")
        self.vocab_size = config.vocab_size
        self.embed_tokens = self.model.embed_tokens

    # @classmethod
    # def init_model(cls, config):
    #     key = id(config)  # use object identity for cache key
    #     if key not in cls._model_cache:
    #         if not isinstance(config, PretrainedConfig):
    #             raise ValueError("config must be an instance of PretrainedConfig")
    #         cls._model_cache[key] = TracLlama3Model(config)
    #     return cls._model_cache[key]
    def init_model(self, config):
        return TracLlama3Model(config)

    def get_model(self):
        return self.model

    # def embed_tokens(self):
    #     return self.model.embed_tokens

    def all_to_device(self, device="cuda"):
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
        inputs_embeds: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        # image_features: Optional[torch.FloatTensor] = None,
        bbox_gts: Optional[torch.FloatTensor] = None,
        bbox_masks: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        enable_bboxes: bool = True,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        pre_input_ids = input_ids

        if inputs_embeds == None:

            (inputs_embeds, labels, image_features) = (
                self.prepare_inputs_for_multimodal(
                    input_ids,
                    images,
                    labels=labels,
                )
            )

        outputs = self.model.forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_masks,
            labels=labels,
            output_hidden_states=output_hidden_states,
            **{k: v for k, v in kwargs.items() if k not in ["input_ids","attention_mask","labels","inputs_embeds"]},
        )

        if enable_bboxes:
            outputs = self._handle_bbox_prediction(
                outputs, image_features, bbox_gts, bbox_masks
            )

        return outputs

    def _handle_bbox_prediction(
        self, outputs, image_features, bbox_gts=None, bbox_masks=None
    ):
        """Handle 3D bounding box prediction"""
        print("handle bbox prediction")

        # Dependency injection
        predictor = self.model.bbox3d_predictor.predict_bboxes
        compute_bbox_loss = self.model.bbox3d_predictor.compute_bbox_loss

        if bbox_masks == None and bbox_gts == None:
            print("no grouth truth or mask, predice bbox mode")
            vision_features = image_features
            text_features = outputs.hidden_states[-1]
            bbox_predictions = predictor(vision_features, text_features)
            outputs["bbox_3d_pred"] = bbox_predictions
        else:
            print("normal mode")

            bbox_samples = bbox_masks.any(dim=1)
            if not bbox_samples.any():
                print("No valid bbox samples found.")
                return outputs

            targets = bbox_gts[bbox_samples]
            masks = bbox_masks[bbox_samples]

            vision_features = image_features[bbox_samples]
            print("vision features", vision_features.shape)
            text_features = outputs.hidden_states[-1][bbox_samples]
            bbox_predictions = predictor(vision_features, text_features)

            bbox_loss = compute_bbox_loss(bbox_predictions, targets, masks)

            # Add to outputs
            outputs.loss = outputs.loss + bbox_loss
            outputs["bbox_3d_loss"] = bbox_loss
            outputs["bbox_3d_pred"] = bbox_predictions

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        bbox3d_enable: bool = False,
        **kwargs,
    ):
        """Generate with optional 3D bbox prediction"""
        # Prepare inputs
        if images is not None:
            (inputs_embeds, _, image_features) = (
                self.prepare_inputs_for_multimodal(input_ids, images, **kwargs)
            )
            kwargs["inputs_embeds"] = inputs_embeds
            kwargs["attention_mask"] = attention_masks
            kwargs["image_features"] = image_features
        forward_output=self.forward(**kwargs)
        kwargs.pop("image_features", None)  
        


        outputs = self.model.generate(
            output_hidden_states=False, return_dict_in_generate=True, **kwargs
        )
        outputs["bbox_3d_pred"] =forward_output["bbox_3d_pred"]
        
        return outputs


    def _process_bbox_generation(self, outputs, images):
        """Process 3D bbox prediction during generation"""
        # Extract hidden states from generation
        last_tensors = [step[-1] for step in outputs.hidden_states]
        last_hidden_state = torch.cat(last_tensors[1:], dim=1)

        bbox_token_mask = (
            outputs.sequences[:, 1:] == self.config.multimodal.bbox3d_token_id
        )

        bbox_prompts = self.model.bbox3d_predictor.extract_bbox_features(
            last_hidden_state, bbox_token_mask
        )

        logits = self.model.bbox3d_predictor.bbox3d_head(images, bbox_prompts)

        no_bbox_ids = [
            i for i, mask in enumerate(bbox_token_mask) if not torch.sum(mask)
        ]
        if no_bbox_ids:
            logits[no_bbox_ids] = -torch.inf

        return logits


from transformers import AutoConfig, AutoModelForCausalLM

# AutoConfig.register("trac-phi3", TracPhi3Config)
# AutoModelForCausalLM.register(TracPhi3Config, TracPhi3ForCausalLM)
if __name__ == "__main__":
    from collator import QA3DDataset, BboxAwareCollator
    from torch.utils.data import DataLoader

    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM

    model_max_length = 512
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    special_tokens = [
        "<im_patch>",
        "<bx_start>",
        "<bx_end>",
        "<image>",
        "<image_newline>",
    ]

    image_token_name = "<im_patch>"
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"Added {num_added} special tokens", len(tokenizer))

    collator = BboxAwareCollator(
        tokenizer=tokenizer,
        max_length=model_max_length,
        max_bbox_length=9,
        num_vision_token=256,
        token_name=image_token_name,
    )

    ds = QA3DDataset()
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collator)
    img_token_id = tokenizer.convert_tokens_to_ids(image_token_name)

    config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    config.img_token_id = img_token_id
    print("before", config.vocab_size)
    config.vocab_size = len(tokenizer)
    model = TracLlamaForCausalLM(config)

    model.get_model().initialize_multimodal_components()
    # model = model.to("cuda")
    model.all_to_device("cuda")
    for i, batch in enumerate(dl):
        (
            images,
            input_ids,
            attention_mask,
            labels,
            bbox_gt,
            bbox_mask,
            position_ids,
        ) = batch.values()
        images = images.to("cuda")
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        labels = labels.to("cuda")
        bbox_gt = bbox_gt.to("cuda")
        bbox_mask = bbox_mask.to("cuda")
        position_ids = position_ids.to("cuda")

        if i == 0:
            print("forward pass")
            print("mask", bbox_mask)
            print("gt", bbox_gt)
            outputs = model(
                input_ids=input_ids,
                images=images,
                bbox_gts=bbox_gt,
                bbox_masks=bbox_mask,
                labels=labels,
                attention_masks=attention_mask,
                position_ids=position_ids,
            )
        elif i == 1:
            print("generation")
            outputs = model.generate(input_ids=input_ids, images=images)
