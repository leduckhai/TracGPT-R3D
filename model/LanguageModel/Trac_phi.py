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
from transformers import AutoConfig, AutoModelForCausalLM, Phi3Config, Phi3Model,PhiModel,PhiConfig,PhiForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from model.LanguageModel.Trac_arch import TracMetaModel, TracMetaForCausalLM

class TracPhi3Config(PhiConfig):
    model_type="trac-phi3"

class TracPhi3Model(TracMetaModel,PhiModel):
    config_class=TracPhi3Config
class TracPhi3ForCausalLM(PhiForCausalLM,TracMetaForCausalLM):
    config_class=TracPhi3Config
    def __init__(self,config):
        # super(TracMetaForCausalLM,self).__init__(config)
        super().__init__(config)
        self.model=TracPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            bboxes3d: Optional[torch.FloatTensor] = None,

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
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )

        try:
            bbox3d_ids = torch.nonzero(torch.sum(bboxes3d, dim=(1, 2))).flatten().tolist()
        except:
            bbox3d_ids = []

        if self.get_model().bbox3d_enable and bbox3d_ids:
            outputs = super().forward(
                                    input_ids=input_ids,
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
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

            bbox3d_token_mask = input_ids_pre[:, 1:] == self.config.bbox3d_token_id
            bbox3d_token_mask = torch.cat(
                [
                    bbox3d_token_mask,
                    torch.zeros((bbox3d_token_mask.shape[0], 1), dtype=bbox3d_token_mask.dtype).cuda(),
                ],
                dim=1,
            )

            bbox3d_prompts = []
            for i in bbox3d_ids:
                if torch.sum(bbox3d_token_mask[i]) == 1:
                    bbox3d_token = last_hidden_state[i][bbox3d_token_mask[i]]
                    bbox3d_prompt = self.get_model().bbox3d_projector(bbox3d_token)
                elif torch.sum(bbox3d_token_mask[i]) > 1:
                    bbox3d_tokens = last_hidden_state[i][bbox3d_token_mask[i]]
                    bbox3d_token = torch.mean(bbox3d_tokens, dim=0, keepdim=True)
                    bbox3d_prompt = self.get_model().bbox3d_projector(bbox3d_token)
                else:
                    bbox3d_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
                                             device=last_hidden_state.device)
                bbox3d_prompts.append(bbox3d_prompt)

            bbox3d_prompts = torch.cat(bbox3d_prompts, dim=0)
            logits = self.get_model().bbox3d_module(images[bbox3d_ids], text_emb=bbox3d_prompts)
            loss_l1 = self.get_model().l1_loss(logits, bboxes3d[bbox3d_ids])
            loss_iou3d = self.get_model().iou3d_loss(logits, bboxes3d[bbox3d_ids])
            bbox3d_loss = loss_l1 + loss_iou3d
            outputs.loss = outputs.loss + bbox3d_loss
            return outputs
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

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
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    config = AutoConfig.from_pretrained("microsoft/phi-2", model_type="trac-phi3")
    config.bbox3d_enable = True 
    config.mm_hidden_size = 512  
    config.bbox3d_token_id = 50295  

    model = TracPhi3ForCausalLM(config)
    model = model.to('cuda')  

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

    # Example inputs
    batch_size = 2
    seq_length = 32
    vocab_size = config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to('cuda')
    attention_mask = torch.ones((batch_size, seq_length)).to('cuda')
    position_ids = torch.arange(seq_length).expand(batch_size, -1).to('cuda')
    images = torch.randn((batch_size, 3, 224, 224)).to('cuda')
    bboxes3d = torch.randn((batch_size, 8, 3)).to('cuda')

    # Text-only generation
    generated_ids = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_length=50
    )

    # Multimodal generation with bbox
    generated_ids, bbox_logits = model.generate(
        inputs=input_ids,
        images=images,
        bbox3d_enable=True,
        max_length=50
    )

    # Print results
    print("=== TEXT-ONLY GENERATION ===")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    print("\n=== MULTIMODAL GENERATION ===")
    print("Generated Text:")
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    print("\n3D Bounding Box Predictions:")
    for i, bbox in enumerate(bbox_logits):
        print(f"\nObject {i+1}:")
        # Format the bbox coordinates nicely
        corners = bbox.cpu().detach().numpy()
        for j, corner in enumerate(corners):
            print(f"Corner {j+1}: x={corner[0]:.2f}, y={corner[1]:.2f}, z={corner[2]:.2f}")