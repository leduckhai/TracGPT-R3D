import torch 
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List
import numpy as np
import torch.nn.functional as F
import sys 
sys.path.append("/root/TracGPT-R3D")
from collections import defaultdict

import torch
from collections import defaultdict

class StandardCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        print("StandardCollator initialized")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        Preprocess a batch of samples for multimodal training.

        Each sample should have:
            - 'image': image tensor [C,H,W]
            - 'Q4': list with question string
            - 'A4': answer string
        Returns:
            dict with:
                - input_ids: [B, seq_len]
                - attention_mask: [B, seq_len]
                - labels: [B, seq_len] (masked question tokens = -100)
                - images: [B, C, H, W]
                - full_texts: list of original text
        """
        images = []
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        full_texts = []

        for sample in batch:
            image=sample['image'].unsqueeze(0)  # [C, H, W]
            images.append(image)
            question = sample['Q4'][0]
            answer = sample['A4']

            text = f"<Question> {question} <Answer> {answer}"
            sample["text"] = text
            full_texts.append(text)

            # Tokenize with padding/truncation to max_length
            tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
            input_ids = tokenized.input_ids[0]           # [seq_len]
            attention_mask = tokenized.attention_mask[0] # [seq_len]

            # Find answer start index
            a_token_ids = self.tokenizer.encode("<Answer>", add_special_tokens=False)
            try:
                answer_start_idx = (input_ids == a_token_ids[0]).nonzero(as_tuple=True)[0].item()
            except IndexError:
                print("Warning: '<Answer>' token not found in input_ids. Using default start index 0.")
                answer_start_idx = 0

            labels = input_ids.clone()
            labels[:answer_start_idx] = -100  # ignore question in loss

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_labels.append(labels)

        processed = {
            "input_ids": torch.stack(batch_input_ids),            # [B, seq_len]
            "attention_mask": torch.stack(batch_attention_masks), # [B, seq_len]
            "labels": torch.stack(batch_labels),                  # [B, seq_len]
            "images": torch.stack(images),                         # [B, C, H, W]
            "full_texts": full_texts
        }

        return processed


if __name__ == "__main__":
    from src.dataset.dataloader import load_data
    train_set, val_set, test_set = load_data(
        train_val_dir="/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/1ad40bdc-da39-4dd8-9d3a-27ce00e754fa/train/data",
        test_dir="/root/TracGPT-R3D/pseudo_3d/32_overlap_slices/1ad40bdc-da39-4dd8-9d3a-27ce00e754fa/test/data",
        dataset="trac_white",
    )
    model="meta-llama/Llama-3.2-1B"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    collator = StandardCollator(tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    for i, batch in enumerate(train_loader):
        if i >= 1:
            break
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        full_texts = batch["full_texts"]
        # status_criteria = batch["status_criteria"]
        print("images shape", images.shape)
        print("input_ids shape", input_ids.shape)
        print("attention_mask shape", attention_mask.shape)
        print("full_texts", full_texts)
        # print("status_criteria", status_criteria)
    # for i, sample in enumerate(test_set):
    #     if i == 3:
    #         break
    #     print(sample.keys())
    #     slice_order = sample["slice_order"]
    #     patient_id = sample["Patient_ID"]
    #     Q1 = sample["Q1"]
    #     A1 = sample["A1"]
    #     Q2 = sample["Q2"]
    #     A2 = sample["A2"]
    #     Q3 = sample["Q3"]
    #     A3 = sample["A3"]
    #     Q4 = sample["Q4"]
    #     A4 = sample["A4"]
    #     print("patient id", patient_id)
    #     # print("slice order", slice_order)
    #     # print("Q1", Q1)
    #     # Q1: bbox

    #     print("A1", A1)
    #     print("len A1", len(A1))
    #     # print("Q2", Q2)
    #     # print("A2", A2)
    #     # print("Q3", Q3)
    #     # print("A3", A3)
    #     print("Q4", Q4)
    #     # Q4 mild dementia,...
    #     print("A4", A4)
    #     image = sample["image"]
    #     print("image shape", image.shape, image.min(), image.max())
    # pass 