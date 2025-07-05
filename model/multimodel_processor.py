import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast


class MultimodalProcessor(nn.Module):
    """Handles the fusion of text and vision inputs for multimodal processing"""

    def __init__(self, vision_encoder, image_token_id: int ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.image_token_id = image_token_id 
        self.ignore_idx = -100  
        print("image token id:", self.image_token_id)

    def prepare_inputs_for_multimodal(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        embed_tokens_fn: Optional[callable] = None,
    ) -> Tuple[
        Optional[torch.LongTensor],
        Optional[torch.FloatTensor],
    ]:
        """
        Prepare inputs for multimodal processing by fusing text and vision inputs.

        Returns:
            input_ids, position_ids, attention_mask, past_key_values,
            inputs_embeds, labels, image_features
        """
        return self._prepare_multimodal_inputs(
            input_ids,
            images,
            labels,
            embed_tokens_fn,
        )

    def _prepare_text_only_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
        embed_tokens_fn: callable,
    ) -> Tuple:
        """Handle text-only inputs"""
        if embed_tokens_fn is not None:
            inputs_embeds = embed_tokens_fn(input_ids)
        else:
            inputs_embeds = None

        return (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            None,
        )

    def _prepare_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        labels: Optional[torch.LongTensor],
        embed_tokens_fn: callable,
    ) -> Tuple:
        """Handle multimodal inputs with vision and text fusion"""
        print("before image encoder", images.shape)
        image_features = self.vision_encoder.encode_images(images)
        print("image features shape:", image_features.shape)
        new_labels = []
        new_inputs_embeds = []

        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            cur_labels = labels[batch_idx] if labels is not None else None
            cur_image_features = image_features[batch_idx : batch_idx + 1]
            
            new_label, new_embed = self._process_single_sample(
                cur_input_ids, cur_labels, cur_image_features, embed_tokens_fn
            )
            if new_label is not None:
                new_labels.append(new_label)
            new_inputs_embeds.append(new_embed)

        return (
            torch.stack(new_inputs_embeds),
            torch.stack(new_labels) if len(new_labels) > 0 else None,
            image_features,
        )

    def _process_single_sample(
        self, input_ids, labels, image_features, embed_tokens_fn=None
    ):
      
        # We expect the image_token line consecuively
        image_token_idx=  (input_ids == self.image_token_id).nonzero(as_tuple=True)[0].tolist()
        if  len(image_token_idx) !=image_features.shape[1]:
            raise ValueError(f"number of image tokens not match  in input_ids expect {image_features.shape[1]} but got {len(image_token_idx)}")

        print("process single sample")
        print("input ids", input_ids.shape,input_ids.dtype,input_ids.min(),input_ids.max())
        text_embed= embed_tokens_fn(input_ids)
        print("image embed shape:", image_features.shape)
        print("text embed shape:", text_embed.shape)
        start_idx = image_token_idx[0]
        end_idx = image_token_idx[-1] + 1
        new_embeds = torch.cat(
            [text_embed[:start_idx], image_features.squeeze(0), text_embed[end_idx:]]
        )
        if labels == None:
            new_labels = None
        else:
            new_labels = torch.cat(
                [
                    labels[:start_idx],
                    torch.full(
                        (image_features.shape[1],),
                        self.ignore_idx,
                        dtype=labels.dtype,
                        device=labels.device,
                    ),
                    labels[end_idx:],
                ]
            )
      
        return  new_labels, new_embeds

    def get_vision_tower(self):
        """Get the vision tower component"""
        return self.vision_encoder.vision_tower

    def get_mm_projector(self):
        """Get the multimodal projector component"""
        return self.vision_encoder.mm_projector


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    import torch.nn as nn

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = ["<image>", "<image_newline>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # model.resize_token_embeddings(len(tokenizer))

    # Now get the actual token IDs
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    image_newline_token_id = tokenizer.convert_tokens_to_ids("<image_newline>")

    class DummyVisionEncoder(nn.Module):
        def encode_images(self, images):
            return torch.randn(images.shape[0], 256, 3072)  # Dummy image features

    vision_encoder = DummyVisionEncoder()
    processor = MultimodalProcessor(vision_encoder,image_token_id=image_token_id)
 

    input_ids = torch.ones(2, 512, dtype=torch.long)

    input_ids[:, 5:5+256] = image_token_id

    images = torch.randn(2, 32, 256, 256)
    labels= torch.ones(2, 512, dtype=torch.long)


    embed_tokens_fn = lambda x: torch.randn(x.shape[0], 3072)
    position_ids = None
    attention_mask = None
    past_key_values = None

    outputs = processor.prepare_inputs_for_multimodal(
        input_ids=input_ids,
        images=images,
        labels=labels,
        embed_tokens_fn=embed_tokens_fn,
    )



