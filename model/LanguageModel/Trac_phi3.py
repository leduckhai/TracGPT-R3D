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
from model.LanguageModel.Trac_arch import TracMetaModel

class TracPhi3Config(Phi3Config):
    model_type="trac-phi3"


class TracPhi3Model(TracMetaModel, Phi3Model):
    config_class = TracPhi3Config

    def __init__(self, config: Phi3Config):
        super(TracMetaModel, self).__init__(config)

class TracPhi3ForCausalLM(Phi3ForCausalLM):

    config_class=TracPhi3Config
    def __init__(self,config):
        super(TracPhi3ForCausalLM, self).__init__(config)
       
        self.model=TracPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def all_to_device(self,device="cuda"):
        baseModel=self.get_model()
        baseModel.to(device)
        if self.vision_tower:
            self.vision_tower.to(device)
        if self.mm_projector:
            self.mm_projector.to(device)

    def get_model(self):
        return self.model
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()



    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_masks: Optional[torch.Tensor] = None,
            bbox_gts: Optional[torch.FloatTensor] = None,
            bbox_masks: Optional[torch.BoolTensor] = None,
            answer_types: Optional[torch.LongTensor] = None,
            
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_pre = input_ids

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_masks,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_masks,
                past_key_values,
                labels,
                images,
            )

      
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        bbox_loss = torch.tensor(0.0, dtype=torch.float, device=device, requires_grad=True)
        if self.get_model().bbox3d_enable and bbox_gts is not None and bbox_masks is not None:
            outputs = super().forward(
                                    input_ids=input_ids,
                                    inputs_embeds=inputs_embeds,
                                    attention_masks=attention_masks,
                                    labels=labels,
                                    output_hidden_states=True,

                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    return_dict=return_dict
                                )

            output_hidden_states = outputs.hidden_states
            last_hidden_state = output_hidden_states[-1]

            # Check if there are any valid bbox samples
            if bbox_masks is not None and bbox_masks.any():
                # Find samples that have at least one valid bbox
                bbox_samples = bbox_masks.any(dim=1)  # [B] - which samples have bboxes
                
                if bbox_samples.any():  # Only proceed if there are bbox samples
                    # Extract features for bbox samples only
                    bbox_vision_features = vision_features[bbox_samples]  # [B_masked, L, D_v]
                    bbox_last_hidden = last_hidden_state[bbox_samples]    # [B_masked, seq_len, hidden_dim]
                    bbox_targets = bbox_gts[bbox_samples]                 # [B_masked, max_bbox_len, 6]
                    bbox_attention_mask = bbox_masks[bbox_samples] if bbox_masks is not None else None
                    
                    # Pool features
                    # For vision features: average over bbox sequence length
                    vision_feat_pooled = bbox_vision_features.mean(dim=1)  # [B_masked, D_v]
                    
                    # For text features: average over sequence length
                    text_feat_pooled = bbox_last_hidden.mean(dim=1)       # [B_masked, hidden_dim]
                    
                    # Combine features
                    combined_features = torch.cat([vision_feat_pooled, text_feat_pooled], dim=-1)
                    
                    # Predict bboxes
                    bbox_pred = self.bbox_3d_head(combined_features)  # Output shape depends on head design
                    
                    # Handle different bbox prediction formats
                    if bbox_pred.dim() == 2 and bbox_targets.dim() == 3:
                        # If prediction is [B_masked, 6] but targets are [B_masked, max_bbox_len, 6]
                        # We need to handle multiple bboxes per sample
                        if bbox_attention_mask is not None:
                            # Use attention mask to compute loss only on valid bboxes
                            valid_mask = bbox_attention_mask[bbox_samples]  # [B_masked, max_bbox_len]
                            
                            # Expand predictions to match target shape
                            bbox_pred_expanded = bbox_pred.unsqueeze(1).expand(-1, bbox_targets.size(1), -1)
                            
                            # Compute loss only on valid bboxes
                            bbox_loss_full = F.smooth_l1_loss(bbox_pred_expanded, bbox_targets, reduction='none')
                            bbox_loss_masked = bbox_loss_full * valid_mask.unsqueeze(-1).float()
                            bbox_loss = bbox_loss_masked.sum() / valid_mask.sum().clamp(min=1)
                        else:
                            # If no attention mask, assume first bbox is the target
                            bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_targets[:, 0, :])
                            
                    elif bbox_pred.dim() == 3 and bbox_targets.dim() == 3:
                        # Both are [B_masked, max_bbox_len, 6]
                        if bbox_attention_mask is not None:
                            valid_mask = bbox_attention_mask[bbox_samples]  # [B_masked, max_bbox_len]
                            bbox_loss_full = F.smooth_l1_loss(bbox_pred, bbox_targets, reduction='none')
                            bbox_loss_masked = bbox_loss_full * valid_mask.unsqueeze(-1).float()
                            bbox_loss = bbox_loss_masked.sum() / valid_mask.sum().clamp(min=1)
                        else:
                            bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_targets)
                            
                    elif bbox_pred.dim() == 2 and bbox_targets.dim() == 2:
                        # Both are [B_masked, 6] - single bbox per sample
                        bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_targets)
                    
                    else:
                        raise ValueError(f"Incompatible shapes: bbox_pred {bbox_pred.shape}, bbox_targets {bbox_targets.shape}")
                    
                    # Store predictions and loss in outputs
                    if hasattr(outputs, '__dict__'):
                        outputs.bbox_3d_loss = bbox_loss
                        outputs.bbox_3d_pred = bbox_pred
                    elif isinstance(outputs, dict):
                        outputs['bbox_3d_loss'] = bbox_loss
                        outputs['bbox_3d_pred'] = bbox_pred
                    return outputs
        
        else:
            return super().forward(
                input_ids=input_ids,
                attention_masks=attention_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )


    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        bbox3d_enable: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_masks = kwargs.pop("attention_masks", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_masks,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_masks,
                None,
                None,
                images,
            )
        else:
            inputs_embeds=super().get_input_embeddings()(inputs)
        if bbox3d_enable:
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs
            )

            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences

            bbox3d_token_mask = output_ids[:, 1:] == self.config.bbox3d_token_id

            last_tensors = [tuple[-1] for tuple in output_hidden_states]
            last_hidden_state = torch.cat(last_tensors[1:], dim=1)

            bbox3d_prompts = []
            nobbox_ids = []
            for i in range(len(bbox3d_token_mask)):
                if torch.sum(bbox3d_token_mask[i]) == 1:
                    bbox3d_token = last_hidden_state[i][bbox3d_token_mask[i]]
                    bbox3d_prompt = self.get_model().bbox3d_projector(bbox3d_token)
                elif torch.sum(bbox3d_token_mask[i]) > 1:
                    bbox3d_tokens = last_hidden_state[i][bbox3d_token_mask[i]]
                    bbox3d_token = torch.mean(bbox3d_tokens, dim=0, keepdim=True)
                    bbox3d_prompt = self.get_model().bbox3d_projector(bbox3d_token)
                else:
                    nobbox_ids.append(i)
                    bbox3d_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                bbox3d_prompts.append(bbox3d_prompt)

            bbox3d_prompts = torch.cat(bbox3d_prompts, dim=0)
            logits = self.get_model().bbox3d_module(images, bbox3d_prompts)
            logits[nobbox_ids] = -torch.inf

            return output_ids, logits
        else:
            output_ids = super().generate(
                inputs_embeds=inputs_embeds,
                **kwargs
            )
            return output_ids


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs


AutoConfig.register("trac-phi3", TracPhi3Config)
AutoModelForCausalLM.register(TracPhi3Config, TracPhi3ForCausalLM)

if __name__=="__main__":
    from collator import QA3DDataset,BboxAwareCollator
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model= TracPhi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model.get_model().to('cuda')

    max_length=512
    collator= BboxAwareCollator(tokenizer=tokenizer)
    ds= QA3DDataset()
    dl=DataLoader(ds, batch_size=2, shuffle=True, collate_fn=BboxAwareCollator(tokenizer=tokenizer,max_length=max_length))

    for batch in dl:

        images= batch['images'].to('cuda')
        input_ids= batch['input_ids'].to('cuda')
        attention_masks = batch['attention_masks'].to('cuda')
        labels= batch['labels'].to('cuda')
        bbox_gts = batch['bbox_3d_gt'].to('cuda')
        bboxe_masks = batch['bbox_3d_mask'].to('cuda')
        answer_types = batch['answer_types'].to('cuda')
        position_ids = batch['position_ids'].to('cuda')
        

        with torch.no_grad():

            generated_ids, bbox_logits = model.generate(
                inputs=input_ids,
                attention_masks=attention_masks,
                position_ids=position_ids,
                images=images,
                bbox3d_enable=True,
                max_length=50
            )

            print("Generated Text:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
            print("3D Bounding Box Predictions:", bbox_logits)

        model.forward(
            input_ids=input_ids,
            attention_masks=attention_masks,
            position_ids=position_ids,
            images=images,
            bboxes3d=bboxes3d
        )
    