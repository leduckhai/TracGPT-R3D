import numpy as np
from tqdm import tqdm
import sys
import os
import pdb
from typing import Tuple, Optional, Union

from peft import LoraConfig, get_peft_model,get_peft_config,PeftModelForCausalLM,TaskType,PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig 

import torch
import torch.nn as nn
from torch.nn import functional as nnf
import sys 
sys.path.append("/home/ubuntu/repo/open-ended-medical-vqa")
import transformers
from transformers import set_seed, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.biogpt import BioGptForCausalLM, BioGptTokenizer, BioGptConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from utils import make_padding

from prefix_mappers import MLP, TransformerMapper


def process_input_image(image):
    pass 


class VQAmedModel(nn.Module):
    # inp torch.Size([32, 512]) torch.Size([32, 37]) torch.Size([32, 50292]) torch.Size([32])
    def process_input(self, inp,mode="train"):
        if mode=="train":
            prefix,question,answer,max_seq_len,patch_img=inp 
            question, mask_question=make_padding(max_seq_len[0],question)
            answer, mask_answer=make_padding(max_seq_len[1],answer)
            
            # return prefix,question,answer,max_seq_len,mask_question,mask_answer,patch_img
        pass 
    def forward(self, prefix, labels, tokens, mask, q_len, batch_size):
        # print("prefix",prefix.shape,"labels",labels.shape,"tokens",tokens.shape,"mask",mask.shape,"q_len",q_len)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # print("self.gpt_embedding_size",self.gpt_embedding_size,"prefix_projections",prefix_projections.shape,"self.prefix_length",self.prefix_length)
        if self.gpttype=='microsoft/biogpt':
          
            embedding = self.gpt.transformer.embed_tokens(tokens)
        else:
    
            embedding = self.gpt.transformer.wte(tokens)
        # print("embed",embedding.shape)
        # embed torch.Size([32, 31, 768])
        for b in range(batch_size):
            # insert the visual prefix after the question 
            
            embedding[b,q_len[b]:q_len[b]+self.prefix_length,:] = prefix_projections[b]  
        return self.gpt(inputs_embeds=embedding, attention_mask=mask)
    def generate(self, prefix, labels, tokens, mask, q_len):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
        if self.gpttype=='microsoft/biogpt':
            embedding_txt = self.gpt.transformer.embed_tokens(tokens)
        else:
            embedding_txt = self.gpt.transformer.wte(tokens)
        embedding_txt[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embedding_txt
    def __init__(
        self,
        prefix_length=2,
        clip_length=2,
        prefix_size=512,
        num_layers=8,
        setting="lora",
        mapping_type="MLP",
        args=None,
    ):
        super(VQAmedModel, self).__init__()
        gpttype = args.model_type
        self.gpttype = gpttype
        if self.gpttype=="microsoft/biogpt":
            self.model_type="biogpt"
        elif self.gpttype=="gpt2-xl":
            self.model_type="gpt2"
        self.setting = setting
        self.prefix_length = prefix_length
        self.tokenizer=AutoTokenizer.from_pretrained(gpttype)
        # print("prefix length",self.prefix_length)
        self.gpt = AutoModelForCausalLM.from_pretrained(gpttype,load_in_8bit=True,device_map='auto')
        # load the relevant fine-tuning strategy 
        if setting == "lora":
            print("lora")
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=='frozen':
            print("frozen")
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == "MLP":
            self.clip_project = MLP((
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                    self.gpt_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")

       

# adaptation of VQAmedModel for ablation studies
class VQAmedModel_abl(nn.Module):
    def forward(self, prefix, labels, tokens, mask, q_len, batch_size,abl):
        embeddings = self.gpt.transformer.wte(tokens)
        if abl=="replace_visual":
            for b in range(batch_size):
                embeddings[b,q_len[b]:q_len[b]+self.prefix_length,:] = self.nv_tokens[b]  
        elif abl=="remove_question":
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
            embeddings[:,q_len[0]:q_len[0]+self.prefix_length,:] = prefix_projections
        elif abl=="swap":
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
            embeddings[:,q_len[0]:q_len[0]+self.prefix_length,:] = prefix_projections
        return self.gpt(inputs_embeds=embeddings, attention_mask=mask)

    def generate(self, prefix, labels, tokens, mask, q_len,abl):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
        embeddings = self.gpt.transformer.wte(tokens)
        if abl=="replace_visual":
            embeddings[q_len:q_len+self.prefix_length,:] = self.nv_tokens[0]  
        elif abl=="remove_question":
            prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
            embeddings[q_len:q_len+self.prefix_length,:] = prefix_projections
        elif abl=="swap":
            prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
            embeddings[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embeddings

    def __init__(
        self,
        prefix_length=2,
        clip_length=2,
        prefix_size=512,
        num_layers=8,
        setting="frozen",
        mapping_type="MLP",
        args=None,
    ):
        super(VQAmedModel_abl, self).__init__()
        gpttype = "gpt2-xl"
        self.model_type = gpttype
        self.setting = setting
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpttype,load_in_8bit=True,device_map='auto')
        if setting == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=='frozen':
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # for the replace_visual ablation study we replace the visual tokens with learnable parameters 
        self.nv_tokens = torch.nn.Parameter(torch.randn(args.batch_size,prefix_length,self.gpt_embedding_size),requires_grad=True).cuda()
        if mapping_type == "MLP":
            self.clip_project = MLP((prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                    self.gpt_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")
if __name__=="__main__":
    import argparse
    import sys 
    import pickle
    sys.path.append("/home/ubuntu/repo/open-ended-medical-vqa")
    # from data_loaders.dataloader import medvqaDataset
    from data_loaders.data_loader_v2 import medvqaDataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--prefix_length", type=int, default=2)
    parser.add_argument("--model_type", type=str, default="gpt2")
    parser.add_argument("--out_dir", type=str, default="output_vqa")
    args = parser.parse_args()

    train_path="/home/ubuntu/repo/SLAKE/train.pkl"
    val_path="/home/ubuntu/repo/SLAKE/val.pkl"
    test_path="/home/ubuntu/repo/SLAKE/test.pkl"
   
    train_dataset = medvqaDataset(train_path,split="train",prefix_length=args.prefix_length,model_type=args.model_type)#,abl=args.ablation)
    prefix, label, tokens, mask, q_len= train_dataset[0]
    print("dype 1",prefix.dtype,"tokens",tokens.dtype,"mask",mask.dtype)
    model= VQAmedModel(
        prefix_length=args.prefix_length,
        clip_length=4,
        # setting=args.setting,
        setting="frozen",
        # mapping_type=args.mapping_type,
        args=args,
    )
    model = model.cuda()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) 
    for  prefix, label, tokens, mask, q_len in tqdm(train_loader):
        
        prefix = prefix.type(torch.float32).to(torch.device("cuda"))
        tokens = tokens.type(torch.long).to(torch.device("cuda"))
        mask = mask.type(torch.long).to(torch.device("cuda"))
        q_len= q_len.type(torch.long).to(torch.device("cuda"))
        
        output=model(prefix, label, tokens, mask, q_len, args.batch_size)
        break
    # val_dataset = medvqaDataset(val_path,split="val",prefix_length=args.prefix_length,model_type=args.model_type)#, abl=args.ablation)
    # test_dataset = medvqaDataset(test_path,split="test",prefix_length=args.prefix_length,model_type=args.model_type,like_test=True)