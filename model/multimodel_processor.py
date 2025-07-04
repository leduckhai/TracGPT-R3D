import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast


class MultimodalProcessor(nn.Module):
    """Handles the fusion of text and vision inputs for multimodal processing"""

    def __init__(self, vision_encoder, image_token_id: int ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.image_token_id = image_token_id  # Will be set during initialization
        # self.image_newline_token_id = None

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
        **kwargs,
    ) -> Tuple[
        torch.LongTensor,
        torch.LongTensor,
        torch.Tensor,
        Optional[List[torch.FloatTensor]],
        torch.FloatTensor,
        Optional[torch.LongTensor],
        Optional[torch.FloatTensor],
    ]:
        """
        Prepare inputs for multimodal processing by fusing text and vision inputs.

        Returns:
            input_ids, position_ids, attention_mask, past_key_values,
            inputs_embeds, labels, image_features
        """
        # Handle text-only case
        if images is None:
            print(" wtf image is none heer ? ")
            return self._prepare_text_only_inputs(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                embed_tokens_fn,
            )

        # input_ids=None
        # position_ids=None
        # attention_mask=None
        # past_key_values=None
        # labels=None

        # return None,None,None,None,None,None,torch.rand(2, 3, 224, 224)
        # Handle multimodal case
        return self._prepare_multimodal_inputs(
            input_ids,
            images,
            position_ids,
            attention_mask,
            past_key_values,
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
        position_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        labels: Optional[torch.LongTensor],
        embed_tokens_fn: callable,
    ) -> Tuple:
        """Handle multimodal inputs with vision and text fusion"""
        print("before image encoder", images.shape)
        image_features = self.vision_encoder.encode_images(images)
        print("image features shape:", image_features.shape)


        if self.image_token_id is None or not (input_ids == self.image_token_id).any():
            raise ValueError("No image tokens found in input_ids")
        
        new_input_ids = []
        new_labels = []
        new_inputs_embeds = []

        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            cur_input_ids = input_ids[batch_idx]
            cur_labels = labels[batch_idx] if labels is not None else None
            cur_image_features = image_features[batch_idx : batch_idx + 1]
            
            # Process this sample
            _, new_label, new_embed = self._process_single_sample(
                cur_input_ids, cur_labels, cur_image_features, embed_tokens_fn
            )

            # new_input_ids.append(new_ids)
            new_labels.append(new_label)
            new_inputs_embeds.append(new_embed)

        max_len = max(seq.shape[0] for seq in new_inputs_embeds)
        padded_labels = []
        padded_embeds = []
        padded_attention_mask = []

        for i in range(batch_size):
            label = new_labels[i]
            embed = new_inputs_embeds[i]
            pad_len = max_len - embed.shape[0]
            if pad_len > 0:                
                if label is not None:
                    padded_label = torch.cat(
                        [
                            label,
                            torch.full(
                                (pad_len,),
                                self.ignore_idx,
                                dtype=label.dtype,
                                device=label.device,
                            ),
                        ]
                    )
                else:
                    padded_label = None
                padded_embed = torch.cat(
                    [
                        embed,
                        torch.zeros(
                            pad_len,
                            embed.shape[1],
                            dtype=embed.dtype,
                            device=embed.device,
                        ),
                    ]
                )

                mask = torch.cat(
                    [
                        torch.ones(embed.shape[0], dtype=torch.bool, device=embed.device),
                        torch.zeros(pad_len, dtype=torch.bool, device=embed.device),
                    ]
                )
            else:
                padded_label = label
                padded_embed = embed
                mask = torch.ones(embed.shape[0], dtype=torch.bool, device=embed.device)

            padded_labels.append(padded_label)
            padded_embeds.append(padded_embed)
            padded_attention_mask.append(mask)

        final_labels = (
            torch.stack(padded_labels) if padded_labels[0] is not None else None
        )
        final_embeds = torch.stack(padded_embeds)
        final_attention_mask = torch.stack(padded_attention_mask)

        if position_ids is not None:
            final_position_ids = (
                torch.arange(
                    max_len, dtype=position_ids.dtype, device=position_ids.device
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        else:
            final_position_ids = None
        if image_features is None:  # Add image features to the end of the sequencem
            print("image feature is None there")
        print("image features shape:", image_features.shape)
        return (
            None,
            final_position_ids,
            final_attention_mask,
            past_key_values,
            final_embeds,
            final_labels,
            image_features,
        )

    def _process_single_sample(
        self, input_ids, labels, image_features, embed_tokens_fn
    ):
        print(
            "image features shape _process_signle:",
            image_features.shape,
            "input_ids shape: _process_signle",
            input_ids.shape,
        )
        # We expect the image_token line consecuively
        image_token_mask = input_ids == self.image_token_id
        image_token_idx= input_ids == self.image_token_id
        print("num tokens:", len(image_token_idx))
        if len(image_token_idx) !=image_features.shape[1]:
            raise ValueError("number of image tokens not match  in input_ids")


        text_embed= embed_tokens_fn(input_ids)
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
      
        return None, new_labels, new_embeds

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
            return torch.randn(images.shape[0], 16, 256)  # Dummy image features

    vision_encoder = DummyVisionEncoder()
    processor = MultimodalProcessor(vision_encoder)
    processor.set_image_tokens(
        image_token_id=image_token_id, image_newline_token_id=image_newline_token_id
    )

    input_ids = torch.tensor([[101, 1001, 989, 102], [101, 500, 102, 0]])

    input_ids[:, 1] = image_token_id

    # Dummy image batch (e.g., for ViT3D)
    images = torch.randn(2, 32, 256, 256)

    # Labels: match input_ids but ignore image token for loss
    labels = input_ids.clone()
    labels[:, 1] = -100

    embed_tokens_fn = lambda x: torch.randn(x.shape[0], 256)
    # Optional fields
    position_ids = None
    attention_mask = None
    past_key_values = None

    outputs = processor.prepare_inputs_for_multimodal(
        input_ids=input_ids,
        images=images,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        labels=labels,
        embed_tokens_fn=embed_tokens_fn,
        # embed_tokens_fn=lambda x: torch.randn(x.shape[0], x.shape[1], 768)  # Dummy embeddings
    )

    # print(outputs)


"""
def _process_single_sample(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        image_features: torch.FloatTensor,
        embed_tokens_fn: callable
    ) -> Tuple[torch.LongTensor, Optional[torch.LongTensor], torch.FloatTensor]:

        print(f"\n[Start] input_ids: {input_ids.tolist()} shape={input_ids.shape}")
        if labels is not None:
            print(f"labels: {labels.tolist()} shape={labels.shape}")
        print(f"image_features shape: {image_features.shape}\n")

        image_token_mask = input_ids == self.image_token_id
        image_token_indices = torch.where(image_token_mask)[0]
        print(f"image_token_indices: {image_token_indices.tolist()}")

        if len(image_token_indices) == 0:
            return input_ids, labels, embed_tokens_fn(input_ids)

        text_embeds = embed_tokens_fn(input_ids)
        print(f"text_embeds.shape = {text_embeds.shape}")  # (seq_len, hidden_dim)

        num_image_tokens = image_features.shape[1]
        image_embeds = image_features.squeeze(0)
        print(f"image_embeds.shape = {image_embeds.shape}")  # (num_image_tokens, hidden_dim)

        new_embeds_list = []
        new_ids_list = []
        new_labels_list = []
        prev_idx = 0

        for img_token_idx in image_token_indices:
            print(f"\n[Image Token @ {img_token_idx}]")
            if img_token_idx > prev_idx:
                print(f"  - Text before image: idx {prev_idx}:{img_token_idx}")
                new_embeds_list.append(text_embeds[prev_idx:img_token_idx])
                new_ids_list.append(input_ids[prev_idx:img_token_idx])
                if labels is not None:
                    new_labels_list.append(labels[prev_idx:img_token_idx])

            new_embeds_list.append(image_embeds)

            img_ids = torch.full((num_image_tokens,), self.image_token_id, dtype=input_ids.dtype)
            new_ids_list.append(img_ids)
            print(f"  - Inserted img_ids: {img_ids.tolist()}")

            if labels is not None:
                img_labels = torch.full((num_image_tokens,), self.ignore_idx, dtype=labels.dtype)
                new_labels_list.append(img_labels)
                print(f"  - Inserted img_labels: {img_labels.tolist()}")

            prev_idx = img_token_idx + 1

        if prev_idx < len(input_ids):
            print(f"\n[Remaining text] from {prev_idx} to end")
            new_embeds_list.append(text_embeds[prev_idx:])
            new_ids_list.append(input_ids[prev_idx:])
            if labels is not None:
                new_labels_list.append(labels[prev_idx:])

        final_embeds = torch.cat(new_embeds_list, dim=0)
        final_ids = torch.cat(new_ids_list, dim=0)
        final_labels = torch.cat(new_labels_list, dim=0) if labels is not None else None

        print(f"\n[Final Output]")
        print(f"final_ids: {final_ids.tolist()} shape={final_ids.shape}")
        if final_labels is not None:
            print(f"final_labels: {final_labels.tolist()} shape={final_labels.shape}")
        print(f"final_embeds.shape = {final_embeds.shape}")

        return final_ids, final_labels, final_embeds

        
[Start] input_ids: [1, 2, 3, 999, 4, 5, 6, 7] shape=torch.Size([8])
labels: [1, 2, 3, -100, 4, 5, 6, 7] shape=torch.Size([8])
image_features shape: torch.Size([1, 2, 4])

image_token_indices: [3]
text_embeds.shape = torch.Size([8, 4])
image_embeds.shape = torch.Size([2, 4])

[Image Token @ 3]
  - Text before image: idx 0:3
  - Inserted img_ids: [999, 999]
  - Inserted img_labels: [-100, -100]

[Remaining text] from 4 to end

[Final Output]
final_ids: [1, 2, 3, 999, 999, 4, 5, 6, 7] shape=torch.Size([9])
final_labels: [1, 2, 3, -100, -100, 4, 5, 6, 7] shape=torch.Size([9])
final_embeds.shape = torch.Size([9, 4])

# question: What is the label
"""
