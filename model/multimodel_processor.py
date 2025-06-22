import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast

class MultimodalProcessor(nn.Module):
    """Handles the fusion of text and vision inputs for multimodal processing"""
    
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.image_token_id = None  # Will be set during initialization
        self.image_newline_token_id = None
        self.ignore_idx = -100
        
    def set_image_tokens(self, image_token_id: int, image_newline_token_id: int = None):
        """Set the special tokens used for image processing"""
        print("Setting image token IDs:")
        print(f"Image token ID: {image_token_id}")
        self.image_token_id = image_token_id
        self.image_newline_token_id = image_newline_token_id
    
    def prepare_inputs_for_multimodal(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        embed_tokens_fn: Optional[callable] = None,
        **kwargs
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, 
               Optional[List[torch.FloatTensor]], torch.FloatTensor, 
               Optional[torch.LongTensor], Optional[torch.FloatTensor]]:
        """
        Prepare inputs for multimodal processing by fusing text and vision inputs.
        
        Returns:
            input_ids, position_ids, attention_mask, past_key_values, 
            inputs_embeds, labels, image_features
        """
        # Handle text-only case
        if images is None:
            return self._prepare_text_only_inputs(
                input_ids, position_ids, attention_mask, 
                past_key_values, labels, embed_tokens_fn
            )
        
        # Handle multimodal case
        return self._prepare_multimodal_inputs(
            input_ids, images, position_ids, attention_mask,
            past_key_values, labels, embed_tokens_fn
        )
    
    def _prepare_text_only_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
        embed_tokens_fn: callable
    ) -> Tuple:
        """Handle text-only inputs"""
        if embed_tokens_fn is not None:
            inputs_embeds = embed_tokens_fn(input_ids)
        else:
            inputs_embeds = None
            
        return (
            input_ids, position_ids, attention_mask, 
            past_key_values, inputs_embeds, labels, None
        )
    
    def _prepare_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
        embed_tokens_fn: callable
    ) -> Tuple:
        """Handle multimodal inputs with vision and text fusion"""
        print("before image encoder",images.shape)
        image_features = self.vision_encoder.encode_images(images)
        print("image features shape:", image_features.shape)
        if image_features is None:
            return self._prepare_text_only_inputs(
                input_ids, position_ids, attention_mask,
                past_key_values, labels, embed_tokens_fn
            )
        
        # Check if we have image tokens in the input
        if self.image_token_id is None or not (input_ids == self.image_token_id).any():
            # No image tokens found, treat as text-only
            return self._prepare_text_only_inputs(
                input_ids, position_ids, attention_mask,
                past_key_values, labels, embed_tokens_fn
            )
        
        # Process each sample in the batch
        new_input_ids = []
        new_labels = []
        new_inputs_embeds = []
        
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            cur_labels = labels[batch_idx] if labels is not None else None
            cur_image_features = image_features[batch_idx:batch_idx+1]
            
            # Process this sample
            new_ids, new_label, new_embed = self._process_single_sample(
                cur_input_ids, cur_labels, cur_image_features, embed_tokens_fn
            )
            
            new_input_ids.append(new_ids)
            new_labels.append(new_label)
            new_inputs_embeds.append(new_embed)
        
        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in new_input_ids)
        
        padded_input_ids = []
        padded_labels = []
        padded_embeds = []
        padded_attention_mask = []
        
        for i in range(batch_size):
            ids = new_input_ids[i]
            label = new_labels[i]
            embed = new_inputs_embeds[i]
            
            # Pad sequences
            pad_len = max_len - ids.shape[0]
            
            if pad_len > 0:
                # Pad input_ids
                padded_ids = torch.cat([
                    ids,
                    torch.zeros(pad_len, dtype=ids.dtype, device=ids.device)
                ])
                
                # Pad labels
                if label is not None:
                    padded_label = torch.cat([
                        label,
                        torch.full((pad_len,), self.ignore_idx, dtype=label.dtype, device=label.device)
                    ])
                else:
                    padded_label = None
                
                # Pad embeddings
                padded_embed = torch.cat([
                    embed,
                    torch.zeros(pad_len, embed.shape[1], dtype=embed.dtype, device=embed.device)
                ])
                
                # Create attention mask
                mask = torch.cat([
                    torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device),
                    torch.zeros(pad_len, dtype=torch.bool, device=ids.device)
                ])
            else:
                padded_ids = ids
                padded_label = label
                padded_embed = embed
                mask = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            
            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_label)
            padded_embeds.append(padded_embed)
            padded_attention_mask.append(mask)
        
        # Stack all sequences
        final_input_ids = torch.stack(padded_input_ids)
        final_labels = torch.stack(padded_labels) if padded_labels[0] is not None else None
        final_embeds = torch.stack(padded_embeds)
        final_attention_mask = torch.stack(padded_attention_mask)
        
        # Update position_ids if needed
        if position_ids is not None:
            final_position_ids = torch.arange(
                max_len, dtype=position_ids.dtype, device=position_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            final_position_ids = None
        
        return (
            final_input_ids, final_position_ids, final_attention_mask,
            past_key_values, final_embeds, final_labels, image_features
        )
    
    def _process_single_sample(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        image_features: torch.FloatTensor,
        embed_tokens_fn: callable
    ) -> Tuple[torch.LongTensor, Optional[torch.LongTensor], torch.FloatTensor]:
        """Process a single sample, replacing image tokens with image features"""
        
        # Find image token positions
        image_token_mask = input_ids == self.image_token_id
        image_token_indices = torch.where(image_token_mask)[0]
        
        if len(image_token_indices) == 0:
            # No image tokens, return as-is
            return input_ids, labels, embed_tokens_fn(input_ids)
        
        # Get text embeddings
        text_embeds = embed_tokens_fn(input_ids)
        
        # Prepare image features for insertion
        num_image_tokens = image_features.shape[1]  # Number of image patches
        image_embeds = image_features.squeeze(0)  # Remove batch dimension
        
        # Process each image token position
        new_embeds_list = []
        new_ids_list = []
        new_labels_list = []
        
        prev_idx = 0
        
        for img_token_idx in image_token_indices:
            # Add text before image token
            if img_token_idx > prev_idx:
                new_embeds_list.append(text_embeds[prev_idx:img_token_idx])
                new_ids_list.append(input_ids[prev_idx:img_token_idx])
                if labels is not None:
                    new_labels_list.append(labels[prev_idx:img_token_idx])
            
            # Add image embeddings
            new_embeds_list.append(image_embeds)
            
            # Create dummy input_ids for image tokens (they won't be used since we have embeddings)
            img_ids = torch.full(
                (num_image_tokens,), 
                self.image_token_id, 
                dtype=input_ids.dtype, 
                device=input_ids.device
            )
            new_ids_list.append(img_ids)
            
            # Handle labels for image tokens
            if labels is not None:
                img_labels = torch.full(
                    (num_image_tokens,), 
                    self.ignore_idx, 
                    dtype=labels.dtype, 
                    device=labels.device
                )
                new_labels_list.append(img_labels)
            
            prev_idx = img_token_idx + 1
        
        # Add remaining text after last image token
        if prev_idx < len(input_ids):
            new_embeds_list.append(text_embeds[prev_idx:])
            new_ids_list.append(input_ids[prev_idx:])
            if labels is not None:
                new_labels_list.append(labels[prev_idx:])
        
        # Concatenate all parts
        final_embeds = torch.cat(new_embeds_list, dim=0)
        final_ids = torch.cat(new_ids_list, dim=0)
        final_labels = torch.cat(new_labels_list, dim=0) if labels is not None else None
        
        return final_ids, final_labels, final_embeds
    
    def _insert_image_newlines(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """Insert newline tokens between image patch rows if needed"""
        if self.image_newline_token_id is None:
            return image_features
        
        # This would reshape image patches to include newline tokens
        # Implementation depends on your specific image patch arrangement
        # For now, return as-is
        return image_features
    
    def get_vision_tower(self):
        """Get the vision tower component"""
        return self.vision_encoder.vision_tower
    
    def get_mm_projector(self):
        """Get the multimodal projector component"""
        return self.vision_encoder.mm_projector

if __name__=="__main__":
    # Example usage
    from transformers import AutoTokenizer
    import torch.nn as nn
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = ["<image>", "<image_newline>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Resize model embeddings to match new vocab size
    # model.resize_token_embeddings(len(tokenizer))

    # Now get the actual token IDs
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_newline_token_id = tokenizer.convert_tokens_to_ids("<image_newline>")

    class DummyVisionEncoder(nn.Module):
        def encode_images(self, images):
            return torch.randn(images.shape[0], 16, 256)  # Dummy image features

    vision_encoder = DummyVisionEncoder()
    processor = MultimodalProcessor(vision_encoder)
    processor.set_image_tokens(image_token_id=image_token_id, image_newline_token_id=image_newline_token_id)
    
    input_ids = torch.tensor([
    [101, 1001, 102],
    [101, 102, 0]  
])
    images = torch.randn(2, 3, 224, 224)  # Dummy images
    position_ids = None
    attention_mask = None
    past_key_values = None
    labels = None

    outputs = processor.prepare_inputs_for_multimodal(
        input_ids=input_ids,
        images=images,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        labels=labels,
        embed_tokens_fn=lambda x: torch.randn(x.shape[0], x.shape[1], 768)  # Dummy embeddings
    )
    
    print(outputs)