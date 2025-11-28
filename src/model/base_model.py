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
    def embed_tokens(self, input_ids):
        return self.lm_model.get_input_embeddings()(input_ids)
    
    def encode_single_image(self, image):
        """Encode a single image with safety checks"""
        image = image.to(self._device)

        if self.freeze_vision_encoder:
            with torch.no_grad():
                image_features = self.vision_encoder(image)
        else:
            image_features = self.vision_encoder(image)
        # print("stats image features",image_features.mean(), image_features.std(), image_features.min(), image_features.max())
        image_features = self.mm_projector(image_features)
        # print("image feature after projector shape", image_features.shape)
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
