import torch
import torch.nn as nn   

class BaseModel:
    def __init__(self,  tokenizer=None):
        print("initializing image processor")
        self.tokenizer = tokenizer
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_token_name = "<image>"
        self.IGNORE_INDEX = -100
        self.IMAGE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(
            self.image_token_name
        )
       
    # def embed_tokens(self, input_ids):
    #     return self.lm_model.get_input_embeddings()(input_ids)
    
    def encode_single_image(self, image):
        """Encode a single image with safety checks"""
        image = image.to(self._device)

        if self.freeze_vision_encoder:
            with torch.no_grad():
                image_features = self.vision_encoder(image)
        else:
            image_features = self.vision_encoder(image)
        image_features = self.mm_projector(image_features)
        return image_features
    
    def load_projector_weight(self, path):
        state_dict = torch.load(path, map_location=self._device)        
        before_load = {name: param.clone() for name, param in self.mm_projector.named_parameters()}
        state_dict = torch.load(path, map_location="cpu")
        self.mm_projector.load_state_dict(state_dict)
        after_load = {name: param.clone() for name, param in self.mm_projector.named_parameters()}
        for name in before_load:
            diff = (before_load[name] - after_load[name]).abs().sum().item()
            print(f"{name}: total change = {diff}")
        print(f"Loaded projector weights from {path}")
  
    def save_projector_weight(self, path):
        torch.save(self.mm_projector.state_dict(), path)
        print(f"Saved projector weights to {path}")
    
    def prepare_input(
    self,
    input_ids,
    images,
    labels=None,
    attention_mask=None,
    position_ids=None,
):
        """
        LLaMA-friendly prepare_input:
        - image_features: [B, Vi, D] from your encode_single_image(images)
        - text_embeds: token embedding for input_ids
        - Replace each IMAGE_TOKEN_ID in input_ids with the Vi image embeddings
        - Keep all text tokens (do NOT drop tokens)
        - Mask labels for image tokens (IGNORE_INDEX)
        - Return position_ids as None (let LLaMA handle RoPE)
        """

        device = input_ids.device
        B, T = input_ids.shape
        embed_dim = self.embed_tokens.weight.shape[1]  # model embedding dim
        # print("images",images.shape)
        # --- encode images -----------------------------------------------------
        image_features = self.encode_single_image(images)  # [B, Vi, D] or [B, D]
        # print("image_features",image_features.shape)
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)  # [B, 1, D]
        Vi = image_features.shape[1]
        assert image_features.shape[2] == embed_dim, "image feat dim must equal LM embed dim"

        # --- masks & labels defaults -------------------------------------------
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device).long()

        if labels is None:
            labels = torch.full_like(input_ids, self.IGNORE_INDEX, device=device)

        # --- token embeddings --------------------------------------------------
        text_embeds = self.embed_tokens(input_ids)  # [B, T, D]

        # Where are image tokens in the sequence? (bool mask)
        image_token_id = torch.tensor(self.IMAGE_TOKEN_ID, device=device, dtype=input_ids.dtype)
        is_image_token = input_ids == image_token_id  # [B, T]

        # Compute output length L: each image token expands to Vi tokens
        num_image_tokens = is_image_token.sum(dim=1)  # [B]
        expanded_lengths = (T - num_image_tokens) + num_image_tokens * Vi
        L = int(expanded_lengths.max().item())

        # Prepare outputs
        input_embeds = torch.zeros((B, L, embed_dim), device=device, dtype=text_embeds.dtype)
        new_labels = torch.full((B, L), self.IGNORE_INDEX, device=device, dtype=labels.dtype)
        new_attention_mask = torch.zeros((B, L), device=device, dtype=attention_mask.dtype)

        for b in range(B):
            out_pos = 0
            img_occurrence_counter = 0
            for t in range(T):
                if is_image_token[b, t]:
                    # choose the appropriate image slice/feature
                    # If there are multiple <IMG> placeholders per sample, assign in order
                    idx = img_occurrence_counter
                    if idx >= Vi:
                        # If more image placeholders than image_features, repeat last or raise
                        idx = Vi - 1
                    img_feats = image_features[b, idx]  # [D]
                    # expand to Vi embeddings (or if image_features[b] is [Vi,D], we may want all Vi)
                    # Most common pattern: use whole image_features[b] (shape [Vi,D]) as consecutive tokens.
                    # Here we append the full Vi tokens for each image token occurrence.
                    # If your design expects each <IMG> to correspond to one of Vi tokens, adapt accordingly.
                    img_block = image_features[b]  # [Vi, D]
                    n = img_block.shape[0]
                    input_embeds[b, out_pos: out_pos + n] = img_block
                    # image tokens have IGNORE_INDEX in labels
                    new_labels[b, out_pos: out_pos + n] = self.IGNORE_INDEX
                    new_attention_mask[b, out_pos: out_pos + n] = 1
                    out_pos += n
                    img_occurrence_counter += 1
                else:
                    # keep text token (always include it)
                    input_embeds[b, out_pos] = text_embeds[b, t]
                    new_labels[b, out_pos] = labels[b, t]
                    new_attention_mask[b, out_pos] = attention_mask[b, t]
                    out_pos += 1

            # If out_pos < L, padding remains (labels=IGNORE_INDEX and mask=0)
            # no need to fill position ids; we'll let the LM create them (position_ids=None)

        # Return position_ids as None for LLaMA so RoPE stays consistent
        position_ids = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)
        return input_embeds, new_labels, new_attention_mask, position_ids

    # def prepare_input(
    #     self,
    #     input_ids,
    #     images,
    #     labels=None,
    #     attention_mask=None,
    #     position_ids=None,
    #     past_key_values=None,
    #     # image_features=None,
    # ):

    #     B, T = input_ids.shape
    #     if images is None:
    #         print("images is None, use text only")
    #         text_embeds = self.embed_tokens(input_ids)
    #         position_ids = torch.arange(
    #             0, T, dtype=torch.long, device=input_ids.device
    #         ).repeat(B, 1)
    #         return text_embeds, labels, attention_mask, position_ids
    #     image_features = self.encode_single_image(images)

    #     if image_features.dim() == 2:  # [B, D]
    #         image_features = image_features.unsqueeze(1)  # [B, 1, D]
    #         Vi = 1
    #     else:
    #         Vi = image_features.shape[1]  # [B, Vi, D]

    #     D = image_features.shape[2]

    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()

    #     if labels is None:
    #         labels = torch.full_like(input_ids, self.IGNORE_INDEX)

    #     text_embeds = self.embed_tokens(input_ids)
    #     image_token_id_tensor = torch.tensor(
    #         self.IMAGE_TOKEN_ID, device=input_ids.device, dtype=input_ids.dtype
    #     )
    #     is_image_token = input_ids == image_token_id_tensor  # [B, T]

    #     num_image_tokens = is_image_token.sum(dim=1)  # [B]
    #     expanded_lengths = (T - num_image_tokens) + num_image_tokens * Vi
    #     L = expanded_lengths.max().item()

    #     input_embeds = torch.zeros(
    #         (B, L, D), device=input_ids.device, dtype=text_embeds.dtype
    #     )
    #     new_labels = torch.full(
    #         (B, L), self.IGNORE_INDEX, device=input_ids.device, dtype=labels.dtype
    #     )
    #     new_attention_mask = torch.zeros(
    #         (B, L), device=input_ids.device, dtype=torch.bool
    #     )
    #     new_position_ids = torch.zeros(
    #         (B, L), device=input_ids.device, dtype=torch.long
    #     )

    #     for b in range(B):
    #         seq_embeds = []
    #         seq_labels = []
    #         seq_masks = []
    #         pos_counter = 0

    #         for t in range(T):
    #             if is_image_token[b, t]:
    #                 seq_embeds.append(image_features[b])  
    #                 seq_labels.append(
    #                     torch.full(
    #                         (Vi,),
    #                         self.IGNORE_INDEX,
    #                         device=labels.device,
    #                         dtype=labels.dtype,
    #                     )
    #                 )
    #                 seq_masks.append(
    #                     torch.ones(Vi, dtype=torch.bool, device=input_ids.device)
    #                 )
    #                 pos_counter += Vi
    #             else:
    #                 if attention_mask[b, t]:
    #                     seq_embeds.append(text_embeds[b, t].unsqueeze(0))
    #                     seq_labels.append(labels[b, t].unsqueeze(0))
    #                     seq_masks.append(
    #                         torch.ones(1, dtype=torch.bool, device=input_ids.device)
    #                     )
    #                     pos_counter += 1

    #         if seq_embeds:
    #             seq_embeds = torch.cat(seq_embeds, dim=0)
    #             seq_labels = torch.cat(seq_labels, dim=0)
    #             seq_masks = torch.cat(seq_masks, dim=0)

    #             current_length = seq_embeds.size(0)
    #             input_embeds[b, :current_length] = seq_embeds
    #             new_labels[b, :current_length] = seq_labels
    #             new_attention_mask[b, :current_length] = seq_masks
    #             new_position_ids[b, :current_length] = torch.arange(
    #                 current_length, device=input_ids.device
    #             )
      
    #     return input_embeds, new_labels, new_attention_mask, new_position_ids
