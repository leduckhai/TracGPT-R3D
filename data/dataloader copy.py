import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import pickle
import sys
import os
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)
from transformers import AutoTokenizer
from data_process.util import convert_list_slice_paths_to_3d
from datasets import load_dataset
import monai.transforms as mtf


import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta
from ..utils.utils import mask2box
from .dataset_info import dataset_info
from .prompt_templates import (
    Caption_templates,
    PosREC_templates,
    PosREG_templates,
    Seg_templates,
)
from .term_dictionary import term_dict
from monai.transforms import Compose, Resize
import numpy as np
from typing import Mapping, Hashable
from data.transform import TrackCrop, NormalizeBBox3D, ResizeBBox3D


class TracDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        model_type,
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
                    keys=["bbox"], orig_size=[50, 256, 412], target_size=[32, 256, 256]
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

            # if self.max_seqs_len[1] < answer_tk.size(0):
            # raise ValueError("answer is longer than max_seqs_len. Consider recalculate the max_seqs_len")
            answer_tk_pad, answer_tk_mask = self.padding(self.max_seqs_len, answer_tk)

            # if len((answer_tk_pad==0).nonzero())!=0:
            #     pad_start = (answer_tk_pad==0).nonzero()[0]
            # else:
            #     pad_start=[]
            # #  This is to place the |endoftext| token right after the contenct of the answer and before the padding
            # if len(pad_start)==0:
            #     answer_tk_pad = torch.cat((answer_tk_pad,word_end_tk))
            # else:
            #     answer_tk_pad = torch.cat((answer_tk_pad[:pad_start],word_end_tk,answer_tk_pad[pad_start:]))

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
        # print("chunk",chunk)
        slice_order = chunk["slice order"]
        patient_id = chunk["Patient ID"]

        image_path = [
            os.path.join(self.image_dir, patient_id, f"{slice}.pkl")
            for slice in slice_order
        ]
        image_3d = convert_list_slice_paths_to_3d(image_path)
        print("image_3d", image_3d.shape)

        q1 = chunk["Q1"][0]
        a1 = chunk["A1"]
        q2 = chunk["Q2"][0]
        a2 = chunk["A2"]
        q3 = chunk["Q3"][0]
        a3 = chunk["A3"]
        q4 = chunk["Q4"][0]
        a4 = chunk["A4"]

        return {
            "image": image_3d,
            "q1": q1,
            "a1": a1,
            "q2": q2,
            "a2": a2,
            "q3": q3,
            "a3": a3,
            "q4": q4,
            "a4": a4,
        }


class ITRDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, "r") as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
            self.data_list = self.data_list[:512]
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split(".")

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if (
                current_tokens + new_tokens_len <= max_tokens
                and random_sentence not in selected_sentences
            ):
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = ".".join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, "r") as text_file:
                    raw_text = text_file.read()
                text = self.truncate_text(raw_text, self.args.max_length)

                text_tensor = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    "image": image,
                    "text": text,
                    "input_id": input_id,
                    "attention_mask": attention_mask,
                    "question_type": "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, "r") as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, "r") as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + " " + answer,
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

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }
                if self.args.seg_enable:
                    ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized

                image = self.transform(image)

                if self.close_ended:
                    question = data["Question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(
                        data["Choice A"],
                        data["Choice B"],
                        data["Choice C"],
                        data["Choice D"],
                    )
                    question = question + " " + choices
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                else:
                    question = data["Question"]
                    answer = str(data["Answer"])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
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

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }

                if self.args.seg_enable:
                    ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQAYNDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_yn_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_yn_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized

                image = self.transform(image)

                question = data["Question"]
                answer = str(data["Answer"])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
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

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }
                if self.args.seg_enable:
                    ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class PosRECDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode="train"):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="test",
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
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif mode == "test":
            self.transform = val_transform

        self.cls_questions = PosREC_templates["cls_questions"]
        self.des_qustions = PosREC_templates["des_questions"]
        self.cls_answers = PosREC_templates["cls_answers"]
        self.des_answers = PosREC_templates["des_answers"]
        self.cls_no_answers = PosREC_templates["cls_no_answers"]
        self.des_no_answers = PosREC_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data["image"]
            seg_path = data["label"]

            image_array = np.load(image_path)  # 1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split("_")[1].split(".")[0])

            try:
                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it["image"]
                seg = it["seg"]  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = (
                    torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                )

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + " " + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.cls_answers).format(box_text)
                    else:
                        question_temple = random.choice(self.des_qustions)
                        question = question_temple.format(
                            random.choice(term_dict[cls_list[cls_id]])
                        )
                        question = self.image_tokens + " " + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.des_answers).format(
                            cls_list[cls_id], box_text
                        )
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.cls_no_answers).format(
                            cls_list[cls_id]
                        )
                    else:
                        question_temple = random.choice(self.des_qustions)
                        question = question_temple.format(
                            random.choice(term_dict[cls_list[cls_id]])
                        )
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.des_no_answers).format(
                            cls_list[cls_id]
                        )

                text_tensor = self.tokenizer(
                    question + " " + answer,
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

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "REC",
                }

                if self.args.seg_enable:
                    ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class PosREGDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode="train"):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="test",
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
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif mode == "test":
            self.transform = val_transform

        self.cls_questions = PosREG_templates["cls_questions"]
        self.des_questions = PosREG_templates["des_questions"]
        self.cls_answers = PosREG_templates["cls_answers"]
        self.des_answers = PosREG_templates["des_answers"]

        self.cls_no_questions = PosREC_templates["cls_questions"]
        self.des_no_questions = PosREC_templates["des_questions"]

        self.cls_no_answers = PosREG_templates["cls_no_answers"]
        self.des_no_answers = PosREG_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data["image"]
            seg_path = data["label"]

            image_array = np.load(image_path)  # 1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split("_")[1].split(".")[0])

            try:
                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)
                image = it["image"]
                seg = it["seg"]  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = (
                    torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                )

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.cls_answers).format(
                            cls_list[cls_id]
                        )
                    else:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.des_answers).format(
                            cls_list[cls_id], random.choice(term_dict[cls_list[cls_id]])
                        )
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_no_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.cls_no_answers).format(
                            cls_list[cls_id]
                        )
                    else:
                        question_temple = random.choice(self.des_no_questions)
                        question = question_temple.format(
                            random.choice(term_dict[cls_list[cls_id]])
                        )
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.des_no_answers).format(
                            cls_list[cls_id]
                        )

                text_tensor = self.tokenizer(
                    question + " " + answer,
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

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "REG",
                }

                if self.args.seg_enable:
                    ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class SegDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=False, mode="train"):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="train",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="test",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f"{tag}.json"),
                is_segmentation=True,
                data_list_key="test",
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
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif mode == "test":
            self.transform = val_transform

        self.cls_questions = Seg_templates["cls_questions"]
        self.des_questions = Seg_templates["des_questions"]
        self.cls_answers = Seg_templates["cls_answers"]
        self.des_answers = Seg_templates["des_answers"]
        self.cls_no_answers = Seg_templates["cls_no_answers"]
        self.des_no_answers = Seg_templates["des_no_answers"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data["image"]
            seg_path = data["label"]

            image_array = np.load(image_path)  # 1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split("_")[1].split(".")[0])

            try:
                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it["image"]
                seg = it["seg"]  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = (
                    torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                )
                if vld_cls:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.cls_answers)
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(
                            random.choice(term_dict[cls_list[cls_id]])
                        )
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.des_answers).format(
                            cls_list[cls_id]
                        )
                else:
                    if not self.description:
                        question_temple = random.choice(self.cls_questions)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.cls_no_answers).format(
                            cls_list[cls_id]
                        )
                    else:
                        question_temple = random.choice(self.des_questions)
                        question = question_temple.format(
                            random.choice(term_dict[cls_list[cls_id]])
                        )
                        question = self.image_tokens + " " + question
                        answer = random.choice(self.des_no_answers).format(
                            cls_list[cls_id]
                        )

                text_tensor = self.tokenizer(
                    question + " " + answer,
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

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "seg": seg,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "seg",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class RefSegDatasetV2(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        print("args.proj_out_num", args.proj_out_num)
        self.image_tokens = "<im_patch>" * args.proj_out_num

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
        set_track_meta(False)

        if mode == "train":
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine="python")
            self.transform = train_transform
        elif mode == "validation":
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine="python")
            self.transform = val_transform
        elif mode == "test":
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine="python")
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.data_root, data["Image"])

                image_array = np.load(image_path)  # 1*32*256*256, normalized

                seg_path = os.path.join(self.args.data_root, data["Mask"])
                seg_array = np.load(seg_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it["image"]
                seg = it["seg"]  # C*D*H*W

                question = data["Question"]
                question = (
                    "<Image Context>" + self.image_tokens + "<Question>" + question
                )

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + "<Answer>" + answer,
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

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "seg": seg,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class RefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

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
        set_track_meta(False)

        if mode == "train":
            self.data_list = pd.read_csv(args.refseg_data_train_path, engine="python")
            self.transform = train_transform
        elif mode == "validation":
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine="python")
            self.transform = val_transform
        elif mode == "test":
            self.data_list = pd.read_csv(args.refseg_data_test_path, engine="python")
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.data_root, data["Image"])

                image_array = np.load(image_path)  # 1*32*256*256, normalized

                seg_path = os.path.join(self.args.data_root, data["Mask"])
                seg_array = np.load(seg_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it["image"]
                seg = it["seg"]  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + " " + question

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + " " + answer,
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

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "seg": seg,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class MultiSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(MultiSegDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        # self.ds_list.append(RefSegDataset(args, tokenizer, mode=mode))
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(
                SegDataset(
                    args, tokenizer, tag=dataset_code, description=False, mode=mode
                )
            )
            self.ds_list.append(
                SegDataset(
                    args, tokenizer, tag=dataset_code, description=True, mode=mode
                )
            )
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MultiPosDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(MultiPosDataset, self).__init__()
        self.tokenizer = tokenizer

        self.dataset_info = dataset_info

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(
                PosRECDataset(
                    args, tokenizer, tag=dataset_code, description=False, mode=mode
                )
            )
            self.ds_list.append(
                PosRECDataset(
                    args, tokenizer, tag=dataset_code, description=True, mode=mode
                )
            )
            self.ds_list.append(
                PosREGDataset(
                    args, tokenizer, tag=dataset_code, description=False, mode=mode
                )
            )
            self.ds_list.append(
                PosREGDataset(
                    args, tokenizer, tag=dataset_code, description=True, mode=mode
                )
            )
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class PosSegDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(PosSegDatasets, self).__init__()
        self.ds_list = [
            MultiPosDataset(args, tokenizer, mode),
            MultiSegDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
            VQAYNDataset(args, tokenizer, mode=mode),
            MultiPosDataset(args, tokenizer, mode),
            # MultiSegDataset(args, tokenizer, mode),
            # MultiSegDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from M3D.configs import M3DConfig

    args = M3DConfig()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    dataset = PosSegDatasets(args, tokenizer, mode="train")
    print(f"Dataset length: {len(dataset)}")
    print(dataset[0])

if __name__ == "__main__":
    import os
    import random
    from sklearn.model_selection import train_test_split

    train_val_dir = "/home/ubuntu/repo/TracGPT-R3D/VLMTrac/50_chunk_data/train"
    patient_records = os.listdir(os.path.join(train_val_dir, "data"))
    patient_records = sorted(patient_records)
    train_records, val_records = train_test_split(
        patient_records, test_size=0.2, random_state=42
    )
    train_set = TracDataset("gpt2", "train", train_records, train_val_dir)
    # image,q1,a1,q2,a2,q3,a3,q4,a4=train_set[0]
    image = train_set[0]["image"]
    q1 = train_set[0]["q1"]
    a1 = train_set[0]["a1"]
    q2 = train_set[0]["q2"]
    a2 = train_set[0]["a2"]
    q3 = train_set[0]["q3"]
    a3 = train_set[0]["a3"]
    q4 = train_set[0]["q4"]
    a4 = train_set[0]["a4"]
    #  (50, 496, 248)
    # print("image",image.shape, "q1",q1.shape,"a1",a1.shape,"q2",q2.shape,"a2",a2.shape,"q3",q3.shape,"a3",a3.shape,"q4",q4.shape,"a4",a4.shape)
    # print("image",image.shape,q1,a1,q2,a2,q3,a3,q4,a4)
# max_len_q1 31 max_len_q2 67 max_len_q3 63 max_len_q4 63
# max_len_a1 584 max_len_a2 231 max_len_a3 30 max_len_a4 27
