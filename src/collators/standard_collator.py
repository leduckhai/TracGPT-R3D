import torch 
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List
import numpy as np
import torch.nn.functional as F
import sys 
sys.path.append("/root/TracGPT-R3D")


class StandardCollator:
    def __init__(self, tokenizer, max_length: int = 512,extra_config:dict=None):
        print("StandardCollator initialized")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_token = "<image>"
        self.IGNORE_INDEX = -100
        self.answer_word_token="answer: "
        self.asnwer_word_tokenized=self.tokenizer.encode(self.answer_word_token,add_special_tokens=False)
        self.pad_token_id=self.tokenizer.pad_token_id 
        if self.pad_token_id is None:
            self.pad_token_id=self.tokenizer.eos_token_id
        print("PAD TOKEN", self.pad_token_id)
        
    def __call__(self, batch):
   
        images = []
        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []
        full_texts = []
        class_labels = []
        p_ids = []

        for sample in batch:
            image = sample['image'].unsqueeze(0)
            images.append(image)
            p_ids.append(sample.get('P_ID', ''))

            question = sample['Q4'][0] if isinstance(sample['Q4'], list) else sample['Q4']
            answer = sample['answer']
            if len(answer)>0:
                answer=answer.strip()+self.tokenizer.eos_token
            answer_text=  answer
            status = sample['A4']
            class_labels.append(status)

            question_text = f"   Question: {question}" + " left_context " +str(self.image_token) + " right_context  answer:" 
            full_text = question_text +   answer_text

            
            sample["text"] = full_text
            full_texts.append(full_text)

            # Tokenize the full text (this should match the labels structure)
            tokenized = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=False
            )
            input_ids = tokenized.input_ids[0]  # [seq_len]
            attention_mask = tokenized.attention_mask[0]  # [seq_len]

            question_tokenized = self.tokenizer(
                question_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                padding=False
            )
            question_length = len(question_tokenized.input_ids[0])
            
            labels = torch.full_like(input_ids, fill_value=self.IGNORE_INDEX)
            
            if len(labels) > question_length:
                labels[question_length:] = input_ids[question_length:].clone()

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
        

        processed = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_masks,
            "labels": batch_labels,
            "images": images_tensor,
            "full_texts": full_texts,
            "class_labels": class_labels,
            "p_ids": p_ids,
        }

        return processed

    def _create_labels(self, input_ids: torch.Tensor, question: str, answer: str) -> torch.Tensor:
        """
        Create labels with proper masking for question and answer tokens.
        """
        labels = input_ids.clone()
        labels[:] = self.IGNORE_INDEX  # Start by ignoring everything
        
        answer_start_idx = self._find_answer_start(input_ids, answer)
        
        if answer_start_idx != -1:
            labels[answer_start_idx:] = input_ids[answer_start_idx:]
        
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = self.IGNORE_INDEX
        
        return labels


    def _find_answer_start(self, input_ids: torch.Tensor, answer: str) -> int:
        """
        Find the start position of the answer in the tokenized sequence.
        """
        if not answer.strip():
            return -1
        
        try:
            answer_token_ids = self.tokenizer.encode(
                answer, 
                add_special_tokens=False,
                return_tensors="pt"
            )[0]
            
            if len(answer_token_ids) == 0:
                return -1
            
            full_seq = input_ids.tolist()
            answer_seq = answer_token_ids.tolist()
            
            # Try to find the answer sequence
            for i in range(len(full_seq) - len(answer_seq) + 1):
                if full_seq[i:i+len(answer_seq)] == answer_seq:
                    return i
            
            first_token = answer_seq[0]
            for i, token_id in enumerate(full_seq):
                if token_id == first_token:
                    return i
                    
            return -1
            
        except Exception as e:
            print(f"Error finding answer start: {e}")
            return -1

        
def find_sublist_index(full_list, sublist):
    """
    Finds the starting index of a sublist within a larger list.
    Returns -1 if not found.
    """
    if not sublist:
        return -1
        
    sublen = len(sublist)
    for i in range(len(full_list) - sublen + 1):
        if full_list[i:i+sublen] == sublist:
            return i
    return -1

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