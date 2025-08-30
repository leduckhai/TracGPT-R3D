import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
import sys
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM
sys.path.append("/root/TracGPT-R3D")
from transformers.generation.utils import GenerationMixin
from src.model.vision_encoder.load_encoder import load_vision_encoder
from src.model.projector.projector import load_mm_projector
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LlamaForCausalLM
from transformers import LlamaConfig

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


# class TracLlamaConfig(LlamaConfig):
class TracConfig(PretrainedConfig):
    model_type = "trac"

    def __init__(self, config=None, **kwargs):
        self.custom_config = config if config is not None else {}
        super().__init__(**kwargs)
        

class LlavaImageProcessor:
    def __init__(self, model, config, tokenizer=None,freeze_vision_encoder=True):
        print("initializing image processor")
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device="cuda"
        self.image_token_name = "<image>"
        self.IGNORE_INDEX = -100
        self.IMAGE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(self.image_token_name)
        
        print("image token id", self.IMAGE_TOKEN_ID)
        self.freeze_vision_encoder = freeze_vision_encoder
        
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
        
        if self.freeze_vision_encoder:
            with torch.no_grad():
                image_features = self.vision_encoder(image)
        else:
            image_features = self.vision_encoder(image)
        
        # print("image features shape", image_features.shape)
    
        image_features = self.mm_projector(image_features)
        # print("image features shape", image_features.shape)
        
        return image_features
    
    def embed_tokens(self, input_ids):
        return self.model.model.embed_tokens(input_ids)
    
    def prepare_input(self, input_ids, images, labels=None, attention_mask=None, 
                     position_ids=None, past_key_values=None):
        
        B, T = input_ids.shape
        if images is None:
            print("images is None, use text only")
            text_embeds = self.embed_tokens(input_ids)
            position_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            return None, text_embeds, labels, attention_mask, position_ids, past_key_values
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
        
        image_token_id_tensor = torch.tensor(self.IMAGE_TOKEN_ID, 
                                           device=input_ids.device, 
                                           dtype=input_ids.dtype)
        is_image_token = (input_ids == image_token_id_tensor)  # [B, T]

        num_image_tokens = is_image_token.sum(dim=1)  # [B]
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


class TracLlavaForCausalLM(GenerationMixin, PreTrainedModel):
    config_class = TracConfig
    
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        cfg = config.custom_config
        
        base_model_name=cfg["language_model"]["name"]
        print("Loading trac base model:", base_model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     base_model_name,
        #     torch_dtype=torch.float16,  
        # )
        self.model = LlamaForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
            )
        
        self.tokenizer = tokenizer
        
        print("Len model before vocab", self.model.config.vocab_size)
        
        self.model.resize_token_embeddings(len(tokenizer))
        print("Len model after vocab", self.model.config.vocab_size)
       
        self.image_processor = LlavaImageProcessor(self.model, cfg, tokenizer)
    def adjust_embeddings_from_num_new_tokens(self, num_new_tokens):
        with torch.no_grad():
            old_embeddings = self.model.get_input_embeddings().weight.data
            old_embeddings_avg = old_embeddings[:-num_new_tokens, :].mean(dim=0)

            new_embeddings = self.model.get_input_embeddings().weight.data
            new_embeddings[-num_new_tokens:, :] = old_embeddings_avg
    def get_input_embeddings(self):
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings()
        elif hasattr(self.model, 'embed_tokens'):
            return self.model.embed_tokens
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        else:
            raise NotImplementedError("Input embeddings not found")
    
    def get_output_embeddings(self):
        if hasattr(self.model, 'get_output_embeddings'):
            return self.model.get_output_embeddings()
        elif hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        else:
            return None 

 
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
        
       
        transformer_outputs = self.model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
        hidden_states = transformer_outputs[0]  # [B, L, H]

        logits = self.model.lm_head(hidden_states)  
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.image_processor.IGNORE_INDEX)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

    

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        images=None,
        attention_mask=None,
        position_ids=None,
        max_new_tokens=128,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        do_sample=False,
        repetition_penalty=1.0,
        **kwargs
    ):
        """
        Custom generate wrapper that preprocesses multimodal input
        and calls HF's generate.
        """
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        # Expand multimodal input if <image> tokens are present
        if images is not None:
            inputs_embeds, labels, attention_mask, position_ids = self.image_processor.prepare_input(
                input_ids=input_ids,
                images=images,
                labels=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None
            )
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
        
        return outputs
    # @torch.no_grad()
    # def generate(
    #     self,
    #     images,
    #     texts,
    #     num_beams=1,
    #     max_new_tokens=20,
    #     min_length=1,
    #     top_p=0.9,
    #     repetition_penalty=1,
    #     length_penalty=1,
    #     temperature=1,
    #     do_sample=False,
    #     stop_words_ids=[2],
    # ):
    #     """
    #     Generate responses for a batch of image-text inputs.
        
    #     Args:
    #         images (torch.Tensor): Batch of pre-processed images.
    #         texts (List[str]): List of text prompts/questions.
    #         For explanations of other parameters, see Hugging Face Transformers documentation for .generate().
            
    #     Returns:
    #         answers (List[str]): List of generated text responses.
    #     """
    #     # 1. Setup stopping criteria (optional)
    #     # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
    #     #     stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])

        
    #     batch_embs = [self.image_processor.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

    #     # 4. Create a batched tensor of embeddings and attention mask
    #     batch_size = len(batch_embs)
    #     max_len = max([emb.shape[1] for emb in batch_embs])
    #     emb_dim = batch_embs[0].shape[2]
    #     dtype = batch_embs[0].dtype
    #     device = batch_embs[0].device

    #     embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
    #     attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
    #     for i, emb in enumerate(batch_embs):
    #         emb_len = emb.shape[1]
    #         embs[i, -emb_len:] = emb[0] # Right-align the sequences
    #         attn_mask[i, -emb_len:] = 1  # Set attention mask to 1 for actual data

    #     # 5. Generate tokens using the underlying language model
    #     with torch.cuda.amp.autocast(enabled=(self.model.dtype == torch.float16)): # Replaces maybe_autocast
    #         outputs = self.llama_model.generate(
    #             inputs_embeds=embs,
    #             attention_mask=attn_mask,
    #             max_new_tokens=max_new_tokens,
    #             num_beams=num_beams,
    #             length_penalty=length_penalty,
    #             temperature=temperature,
    #             do_sample=do_sample,
    #             min_length=min_length,
    #             top_p=top_p,
    #             repetition_penalty=repetition_penalty,
    #             # stopping_criteria=stopping_criteria, # Now using the criteria
    #             pad_token_id=self.llama_tokenizer.pad_token_id,
    #             eos_token_id=self.llama_tokenizer.eos_token_id,
    #         )

    #     # # 6. Decode and clean up the generated outputs
    #     answers = []
    #     for output_token in outputs:
    #         if output_token[0] == 0:  # Sometimes a batch padding token (0) is at the start
    #             output_token = output_token[1:]
    #         # Decode the token IDs to text
    #         output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
    #         # Clean up the output: keep only the model's answer
    #         # output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
    #         # output_texts = output_texts.replace("<s>", "") # remove any initial start tokens
    #         # output_texts = output_texts.split(r'[/INST]')[-1].strip() # Extract text after the last instruction prompt
    #         answers.append(output_texts)
 
    #     return answers

if __name__ == "__main__":
    import sys 
    

    from src.dataset.dataloader import load_data
    from src.collators.load_collator import load_collator
    import yaml
    
    # config_path="/root/TracGPT-R3D/config/vit_llama_3B.yaml"
    # config_path="/root/TracGPT-R3D/config/vit_phi1B.yaml"
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
    print("pad token id:", tokenizer.pad_token_id)
    
    config=TracConfig(custom_config)
    model = TracLlavaForCausalLM(config,tokenizer=tokenizer)
    model.adjust_embeddings_from_num_new_tokens(len(new_tokens))
    model.to("cuda")
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nAfter modification:")
    print("pad_token:", tokenizer.pad_token)
    print("padding_side:", tokenizer.padding_side)
    print("vocab_size:", len(tokenizer))
    
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
        output=model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images
        )
        break
    for batch in test_loader:
        print("test loader mode")
        input_ids = batch["input_ids"].to(device)
        print("decoded input_ids", tokenizer.batch_decode(input_ids, skip_special_tokens=True))
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        images = batch["images"].to(device)
        
        text=batch["full_texts"]
        # print("text", text) 
        answer=batch["class_labels"]
      
        with torch.inference_mode():
            prompt = "Question: What is the symtompt? "
            inputs = tokenizer(prompt, return_tensors="pt")

            # Generate
            input_ids=inputs.input_ids
            attention_mask=inputs.attention_mask
            input_ids=input_ids.to("cuda")
            attention_mask=attention_mask.to("cuda")
            
            # generate_ids = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_new_tokens=50,
            #     pad_token_id=tokenizer.pad_token_id,)
            
            generate_ids = model.generate(
                input_ids=input_ids,
                # images=images,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,)
            
            
            output=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print("output", output)
            # prompt_texts = batch["full_texts"]   
            # device="cuda"

            # decode_input=tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            # print("decode_input", decode_input)
            # # print("input_ids", input_ids)
            # # print("attention_mask", attention_mask)
            # outputs = model.generate(
            #     # images=images,
            #     input_ids=input_ids,
            #     # max_new_tokens=50,
            #     attention_mask=attention_mask,
            #     # temperature=0.7,
            #     # top_p=0.9
            # )
            
            # text=tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # # print("Generated text:", text)
            # for i in range(len(text)):
            #     print("Generated text:", text[i])
            #     # print("Answer:", answer[i])
        break
    
    
        