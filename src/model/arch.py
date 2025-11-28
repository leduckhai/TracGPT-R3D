from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss


class TracMetaModel:
    def __init__(self, config):
        super(TracMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)


        if model_args.pretrain_vision_model is not None:
            vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=True)

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

class TracMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_input(
        self,
        input_ids,
        images,
        labels=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
    ):

        B, T = input_ids.shape
        if images is None:
            print("images is None, use text only")
            text_embeds = self.embed_tokens(input_ids)
            position_ids = torch.arange(
                0, T, dtype=torch.long, device=input_ids.device
            ).repeat(B, 1)
            return text_embeds, labels, attention_mask, position_ids
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

        text_embeds = self.embed_tokens(input_ids)
        image_token_id_tensor = torch.tensor(
            self.IMAGE_TOKEN_ID, device=input_ids.device, dtype=input_ids.dtype
        )
        is_image_token = input_ids == image_token_id_tensor  # [B, T]

        num_image_tokens = is_image_token.sum(dim=1)  # [B]
        expanded_lengths = (T - num_image_tokens) + num_image_tokens * Vi
        L = expanded_lengths.max().item()

        input_embeds = torch.zeros(
            (B, L, D), device=input_ids.device, dtype=text_embeds.dtype
        )
        new_labels = torch.full(
            (B, L), self.IGNORE_INDEX, device=input_ids.device, dtype=labels.dtype
        )
        new_attention_mask = torch.zeros(
            (B, L), device=input_ids.device, dtype=torch.bool
        )
        new_position_ids = torch.zeros(
            (B, L), device=input_ids.device, dtype=torch.long
        )

        for b in range(B):
            seq_embeds = []
            seq_labels = []
            seq_masks = []
            pos_counter = 0

            for t in range(T):
                if is_image_token[b, t]:
                    seq_embeds.append(image_features[b])  
                    seq_labels.append(
                        torch.full(
                            (Vi,),
                            self.IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype,
                        )
                    )
                    seq_masks.append(
                        torch.ones(Vi, dtype=torch.bool, device=input_ids.device)
                    )
                    pos_counter += Vi
                else:
                    if attention_mask[b, t]:
                        seq_embeds.append(text_embeds[b, t].unsqueeze(0))
                        seq_labels.append(labels[b, t].unsqueeze(0))
                        seq_masks.append(
                            torch.ones(1, dtype=torch.bool, device=input_ids.device)
                        )
                        pos_counter += 1

            if seq_embeds:
                seq_embeds = torch.cat(seq_embeds, dim=0)
                seq_labels = torch.cat(seq_labels, dim=0)
                seq_masks = torch.cat(seq_masks, dim=0)

                current_length = seq_embeds.size(0)
                input_embeds[b, :current_length] = seq_embeds
                new_labels[b, :current_length] = seq_labels
                new_attention_mask[b, :current_length] = seq_masks
                new_position_ids[b, :current_length] = torch.arange(
                    current_length, device=input_ids.device
                )
      
        return input_embeds, new_labels, new_attention_mask, new_position_ids

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")