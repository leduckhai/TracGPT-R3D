import torch 
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List
import numpy as np
import torch.nn.functional as F
import sys 
sys.path.append("/root/TracGPT-R3D")
import random

class StandardCollator:
    def __init__(self, tokenizer, mode="train"):
        print("StandardCollator initialized")
        self.tokenizer = tokenizer
        self.mode = mode        
        self.image_token = "<image>"
        self.IGNORE_INDEX = -100
        self.answer_word_text = "|answer|"
        self.answer_word_token=self.tokenizer(
                self.answer_word_text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                add_special_tokens=False
            )
        self.pad_token_id=self.tokenizer.pad_token_id 
        if self.pad_token_id is None:
            self.pad_token_id=self.tokenizer.eos_token_id
        print("PAD TOKEN", self.pad_token_id)
        
    def __call__(self, batch):
        images, batch_input_ids, batch_attention_masks, batch_labels = [], [], [], []
        full_texts, class_labels, p_ids,question_texts = [], [], [],[]

        for sample in batch:
            image = sample["image"]
            images.append(image)
            p_ids.append(sample.get("P_ID", ""))
            idx=random.randrange(len(sample["Q4"]))
            question=sample["question"]
            answer = sample["answer"].strip()
            status = sample["A4"]
            class_labels.append(status)
            question_text = f"|Question|: {question}  {self.image_token}  "
            question_texts.append(question)
            full_text=question_text + self.answer_word_text 
            question_token=self.tokenizer(
                question_text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            if len(answer) and self.mode == "train":
                answer += self.tokenizer.eos_token
                full_text = full_text + " " + answer
                answer_token=self.tokenizer(
                    answer,
                    return_tensors="pt",
                    truncation=True,
                    padding=False,
                    add_special_tokens=False,
                )
           
                input_ids=torch.cat([question_token.input_ids[0],self.answer_word_token.input_ids[0],answer_token.input_ids[0]],dim=0)
                attention_mask=torch.cat([question_token.attention_mask[0],self.answer_word_token.attention_mask[0],answer_token.attention_mask[0]],dim=0)
            else:
                input_ids=torch.cat([question_token.input_ids[0],self.answer_word_token.input_ids[0]],dim=0)
                attention_mask=torch.cat([question_token.attention_mask[0],self.answer_word_token.attention_mask[0]],dim=0)
            
            full_texts.append(full_text)
            
            # answer_idx=torch.where(input_ids==self.answer_word_token[0])[0]
            # if len(answer_idx)>0:
            #     q_len=answer_idx[0]+1
            q_len=len(question_token.input_ids[0])+len(self.answer_word_token.input_ids[0])  
            labels = torch.full_like(input_ids, fill_value=self.IGNORE_INDEX)
            # if len(labels) > q_len:
            labels[q_len:] = input_ids[q_len:].clone()

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_labels.append(labels)

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        batch_attention_masks = torch.nn.utils.rnn.pad_sequence(
            batch_attention_masks, batch_first=True, padding_value=0
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=self.IGNORE_INDEX
        )
        images_tensor = torch.stack(images)  
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_masks,
            "labels": batch_labels,
            "images": images_tensor,
            "full_texts": full_texts,
            "class_labels": class_labels,
            "question_texts": question_texts,   
            "p_ids": p_ids,
        }
        
if __name__ == "__main__":
    from src.dataset.dataloader import load_data
    train_set, val_set, test_set = load_data(
        train_val_dir="pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/train/data",
        test_dir="pseudo_3d/32_overlap_slices/820ac9e9-3f29-498d-b717-466d44081411/test/data",
        image_train_path="clean_data/train/image",
        image_test_path="clean_data/test/image",
        dataset="trac_white",
    )
    model="meta-llama/Llama-3.2-1B"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    collator = StandardCollator(tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=3,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=3,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    for i, batch in enumerate(val_loader):
        if i >= 3:
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