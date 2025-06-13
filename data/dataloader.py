import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import pickle
import sys
import os
from dotenv import load_dotenv
from types import SimpleNamespace
load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
from transformers import AutoTokenizer
from data_process.util import convert_list_slice_paths_to_3d
import monai.transforms as mtf
import random
import os
import numpy as np
import json
from monai.transforms import Compose, ResizeD
from typing import Mapping, Hashable
from data.transform import TrackCrop, NormalizeBBox3D, ResizeBBox3D
from collections import defaultdict

class TracDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        mode,
        patient_records,
        root_dir,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.base_transform = Compose(
            [
                ResizeD(keys=["image"], spatial_size=[32, 256, 256], mode="bilinear"),
                ResizeBBox3D(
                    keys=["bboxes"], orig_size=[50, 256, 412], target_size=[32, 256, 256]
                ),
            ]
        )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        self.root_dir = os.path.join(ROOT, "VLMTrac/50_chunk_data", mode)
        self.image_dir = os.path.join(self.root_dir, "image")
        self.data_dir = os.path.join(self.root_dir, "data")
        self.data_paths = [
            os.path.join(self.data_dir, record) for record in patient_records
        ]
        self.chunk_data = []
        for path in self.data_paths:
            with open(path, "r") as f:
                data = json.load(f)
                self.chunk_data.extend(data)
        self.max_seqs_len = 100
        self.prefix_len = 300

    def __len__(self):
        return len(self.chunk_data)

    def padding(self, max_seq_len, tokens):
        padding = max_seq_len - tokens.size(0)
        if padding > 0:

            tokens = torch.cat((tokens, torch.zeros(padding)))
            mask = torch.cat(torch.ones(tokens.size(0), torch.zeros(max_seq_len)))

        elif padding == 0:
            mask = torch.ones(tokens.size(0))

        elif padding < 0:

            tokens = tokens[:max_seq_len]
            mask = torch.ones(max_seq_len)
        return tokens, mask

    def process_input_text(self, question, answer=""):
        word_question_tk = torch.tensor(self.tokenizer.encode("question: "))
        word_context_tk = torch.tensor(self.tokenizer.encode(" context:"))
        word_answer_tk = torch.tensor(self.tokenizer.encode("answer "))
        word_end_tk = torch.tensor(self.tokenizer.encode("|endoftext|"))

        mask_word_question_tk = torch.ones(len(word_question_tk))
        mask_word_context_tk = torch.ones(len(word_context_tk))
        mask_word_answer_tk = torch.ones(len(word_answer_tk))
        mask_word_end_tk = torch.zeros(len(word_end_tk))

        if self.mode == "train":
            # construct the model input. The order is question, image, answer. During training the answer is masked. Any padding is placed on the right of the sequence.
            # placeholder tokens are used on the location where the visual prefix will be inserted, with q_len indicating this location.
            #
            question_tk = torch.tensor(self.tokenizer.encode(question))
            answer_tk = torch.tensor(self.tokenizer.encode(answer))

            question_tk_pad, question_tk_mask = self.padding(
                self.max_seqs_len, question_tk
            )
            question_len = (
                word_question_tk.size(0) + question_tk.size(0) + word_context_tk.size(0)
            )

         
            answer_tk_pad, answer_tk_mask = self.padding(self.max_seqs_len, answer_tk)

            question_tk_pad = torch.cat(
                (
                    word_question_tk,
                    question_tk_pad,
                    word_context_tk,
                    torch.ones(self.prefix_len),
                    word_answer_tk,
                    answer_tk_pad,
                )
            )
            question_tk_mask = torch.cat(
                (
                    mask_word_question_tk,
                    question_tk_mask,
                    mask_word_context_tk,
                    torch.ones(self.prefix_len),
                    mask_word_answer_tk,
                    answer_tk_mask,
                    mask_word_end_tk,
                )
            )
            return question_tk_pad, question_tk_mask, question_len
        else:
            # in the test stage we do not have acces to the answer, so we just load the question.
            # since inference is not performed batch-wised we don't need to apply padding

            question_tk = torch.tensor(self.tokenizer.encode(question))

            question_tk_pad, question_tk_mask, _ = self.padding(
                self.max_seqs_le, question_tk
            )
            question_len = (
                word_question_tk.size(0) + question_tk.size(0) + word_context_tk.size(0)
            )
            question_tk_pad = torch.cat(
                (
                    word_question_tk,
                    question_tk_pad,
                    word_context_tk,
                    torch.ones(self.prefix_len),
                    word_answer_tk,
                )
            )

            question_mask = torch.cat(
                (
                    mask_word_question_tk,
                    question_tk_mask,
                    mask_word_context_tk,
                    torch.ones(self.prefix_len),
                    mask_word_answer_tk,
                )
            )
            return question_tk_pad, question_mask, question_len

    def __getitem__(self, idx):
        chunk = self.chunk_data[idx]
        slice_order = chunk["slice order"]
        patient_id = chunk["Patient ID"]

        image_path = [
            os.path.join(self.image_dir, patient_id, f"{slice}.pkl")
            for slice in slice_order
        ]
        image_3d = convert_list_slice_paths_to_3d(image_path)

        q1 = chunk["Q1"][0]
        a1 = chunk["A1"]
        bboxes=a1 
                
        q2 = chunk["Q2"][0]
        a2 = chunk["A2"]
        q3 = chunk["Q3"][0]
        a3 = chunk["A3"]
        q4 = chunk["Q4"][0]
        a4 = chunk["A4"]
        item={
            "image": image_3d,
            "bboxes": bboxes,
        }
        transform_item = self.base_transform(item)
        image= transform_item["image"]
        bboxes = transform_item["bboxes"]
        print("after base transform", image.shape, bboxes) 
        data=defaultdict(list)

        for i,(q,a) in enumerate(zip([q1, q2, q3, q4], [a1, a2, a3, a4])):
            question="<Image Context>"+self.image_tokens+"<Question>"+q
            
            input=question+"<Answer>" + str(a)
            self.tokenizer.padding_side = "right"
            text_tensor = self.tokenizer(
                input,
                max_length=self.args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            question_tensor= self.tokenizer(
                question,
                max_length=self.args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]
            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id
            question_len = torch.sum(question_tensor["attention_mask"][0])
            label = input_id.clone()
            label[:question_len] = -100

            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                label[label == self.tokenizer.pad_token_id] = -100
                if valid_len < len(label):
                    label[valid_len] = self.tokenizer.eos_token_id
            else:
                label[label == self.tokenizer.pad_token_id] = -100
            if i==0:
                data["require_bbox"].append({
                "image":image,
                "bboxes": bboxes,
                "input_id":input_id,
                "label": label,
                "attention_mask": attention_mask,
                })
            else:
                data["no_require_bbox"].append({
                "image": image,
                "input_id": input_id,
                "label": label,
                "attention_mask": attention_mask,
                })

        
        return data



if __name__ == "__main__":
    import os
    import random
    from sklearn.model_selection import train_test_split
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    train_val_dir = "/home/ubuntu/repo/TracGPT-R3D/VLMTrac/50_chunk_data/train"
    patient_records = os.listdir(os.path.join(train_val_dir, "data"))
    patient_records = sorted(patient_records)
    train_records, val_records = train_test_split(
        patient_records, test_size=0.2, random_state=42
    )

    config=SimpleNamespace(
        max_length=512,
        proj_out_num=256
    )

    train_set = TracDataset(args=config, tokenizer=tokenizer,mode="train",patient_records= train_records,root_dir= train_val_dir)
    # image,q1,a1,q2,a2,q3,a3,q4,a4=train_set[0]
    sapmle= train_set[0]
    for key in sapmle:
        print(key, len(sapmle[key]))
        for item in sapmle[key]:
            print(item["image"].shape, item["input_id"].shape, item["label"].shape, item["attention_mask"].shape)
    #  (50, 496, 248)
    # print("image",image.shape, "q1",q1.shape,"a1",a1.shape,"q2",q2.shape,"a2",a2.shape,"q3",q3.shape,"a3",a3.shape,"q4",q4.shape,"a4",a4.shape)
    # print("image",image.shape,q1,a1,q2,a2,q3,a3,q4,a4)
# max_len_q1 31 max_len_q2 67 max_len_q3 63 max_len_q4 63
# max_len_a1 584 max_len_a2 231 max_len_a3 30 max_len_a4 27
