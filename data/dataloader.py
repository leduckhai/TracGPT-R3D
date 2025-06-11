import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json 
import pickle
import sys 
sys.path.append("/home/ubuntu/repo/TracGPT-R3D")
from transformers import AutoTokenizer
from data_process.util import convert_list_slice_paths_to_3d
class TracDataset(Dataset):
    def __init__(self,model_type, mode,patient_records,root_dir, transform=None):
        self.root_dir="/home/ubuntu/repo/TracGPT-R3D/VLMTrac/50_chunk_data"
        self.root_dir=os.path.join(self.root_dir,mode)
        self.image_dir=os.path.join(self.root_dir,"image")
        self.data_dir=os.path.join(self.root_dir,"data")
        self.tokenizer= AutoTokenizer.from_pretrained(model_type)
        self.mode = mode
        self.data_paths=   [os.path.join(self.data_dir,record) for record in patient_records]
        self.chunk_data=[]
        for path in self.data_paths:
            with open(path,"r") as f:
                data=json.load(f)
                self.chunk_data.extend(data)
        print("len",len(self.chunk_data))
        self.max_seqs_len=100
        self.prefix_len=300
     
    def __len__(self):
        return len(self.chunk_data)
    def padding(self,max_seq_len,tokens):
        padding = max_seq_len - tokens.size(0) 
        if padding > 0:
            
            tokens = torch.cat((tokens, torch.zeros(padding)))
            mask = torch.cat(torch.ones(tokens.size(0),torch.zeros(max_seq_len) ))   
              
        elif padding==0:
            mask=torch.ones(tokens.size(0))
        
        elif padding < 0:
        
            tokens = tokens[:max_seq_len]
            mask = torch.ones(max_seq_len)
        return tokens, mask
    

    def process_input_text(self,question,answer=""):
        word_question_tk=torch.tensor(self.tokenizer.encode('question: '))
        word_context_tk=torch.tensor(self.tokenizer.encode(' context:'))
        word_answer_tk=torch.tensor(self.tokenizer.encode('answer '))
        word_end_tk=torch.tensor(self.tokenizer.encode('|endoftext|'))

        mask_word_question_tk=torch.ones(len(word_question_tk))
        mask_word_context_tk=torch.ones(len(word_context_tk))
        mask_word_answer_tk=torch.ones(len(word_answer_tk))
        mask_word_end_tk=torch.zeros(len(word_end_tk))

        if self.mode=="train":
                # construct the model input. The order is question, image, answer. During training the answer is masked. Any padding is placed on the right of the sequence. 
                # placeholder tokens are used on the location where the visual prefix will be inserted, with q_len indicating this location. 
                # 
                question_tk=torch.tensor(self.tokenizer.encode(question))
                answer_tk=torch.tensor(self.tokenizer.encode(answer))

                question_tk_pad, question_tk_mask = self.padding(self.max_seqs_len,question_tk)
                question_len=word_question_tk.size(0) + question_tk.size(0) + word_context_tk.size(0)
                
                # if self.max_seqs_len[1] < answer_tk.size(0):
                    # raise ValueError("answer is longer than max_seqs_len. Consider recalculate the max_seqs_len")
                answer_tk_pad, answer_tk_mask = self.padding(self.max_seqs_len,answer_tk)
        
            
                # if len((answer_tk_pad==0).nonzero())!=0:
                #     pad_start = (answer_tk_pad==0).nonzero()[0]
                # else:
                #     pad_start=[]
                # #  This is to place the |endoftext| token right after the contenct of the answer and before the padding
                # if len(pad_start)==0:
                #     answer_tk_pad = torch.cat((answer_tk_pad,word_end_tk))
                # else:
                #     answer_tk_pad = torch.cat((answer_tk_pad[:pad_start],word_end_tk,answer_tk_pad[pad_start:]))

                question_tk_pad= torch.cat((word_question_tk,question_tk_pad,word_context_tk,torch.ones(self.prefix_len),word_answer_tk,answer_tk_pad))                
                question_tk_mask = torch.cat((mask_word_question_tk,question_tk_mask,mask_word_context_tk,torch.ones(self.prefix_len),mask_word_answer_tk,answer_tk_mask,mask_word_end_tk))
                return question_tk_pad,question_tk_mask,question_len
        else:
                # in the test stage we do not have acces to the answer, so we just load the question. 
                # since inference is not performed batch-wised we don't need to apply padding
                        
                question_tk=torch.tensor(self.tokenizer.encode(question))
                
                question_tk_pad,question_tk_mask,_=self.padding(self.max_seqs_le,question_tk)
                question_len=word_question_tk.size(0)+question_tk.size(0)+word_context_tk.size(0)
                question_tk_pad=torch.cat((word_question_tk,question_tk_pad,word_context_tk,torch.ones(self.prefix_len),word_answer_tk))
                            
                question_mask=torch.cat((mask_word_question_tk,question_tk_mask,mask_word_context_tk,torch.ones(self.prefix_len),mask_word_answer_tk))
                return question_tk_pad,question_mask,question_len

    def __getitem__(self, idx):
        chunk=self.chunk_data[idx]  
        # print("chunk",chunk)
        slice_order=chunk["slice order"] 
        patient_id=chunk["Patient ID"]   

        
        image_path = [os.path.join(self.image_dir, patient_id,f'{slice}.pkl') for slice in slice_order]
        image_3d=convert_list_slice_paths_to_3d(image_path)
        print("image_3d",image_3d.shape)
        
        q1=chunk["Q1"][0]
        a1=chunk["A1"]
        q2=chunk["Q2"][0]
        a2=chunk["A2"]
        q3=chunk["Q3"][0]
        a3=chunk["A3"]
        q4=chunk["Q4"][0]
        a4=chunk["A4"]
        
        return{
        "image":image_3d,
        "q1":q1,
        "a1":a1,
        "q2":q2,
        "a2":a2,
        "q3":q3,
        "a3":a3,
        "q4":q4,
        "a4":a4,

        } 

if __name__=="__main__":
    import os
    import random
    from sklearn.model_selection import train_test_split
    train_val_dir="/home/ubuntu/repo/TracGPT-R3D/VLMTrac/50_chunk_data/train"
    patient_records=os.listdir(os.path.join(train_val_dir,"data"))
    patient_records = sorted(patient_records) 
    train_records, val_records = train_test_split(patient_records, test_size=0.2, random_state=42)
    train_set=TracDataset("gpt2","train",train_records,train_val_dir)
    # image,q1,a1,q2,a2,q3,a3,q4,a4=train_set[0]
    image=train_set[0]["image"]
    q1=train_set[0]["q1"]
    a1=train_set[0]["a1"]
    q2=train_set[0]["q2"]
    a2=train_set[0]["a2"]
    q3=train_set[0]["q3"]
    a3=train_set[0]["a3"]
    q4=train_set[0]["q4"]
    a4=train_set[0]["a4"]

    print("trainset 0",train_set[0])
    # print("image",image.shape, "q1",q1.shape,"a1",a1.shape,"q2",q2.shape,"a2",a2.shape,"q3",q3.shape,"a3",a3.shape,"q4",q4.shape,"a4",a4.shape)
    # print("image",image.shape,q1,a1,q2,a2,q3,a3,q4,a4)
# max_len_q1 31 max_len_q2 67 max_len_q3 63 max_len_q4 63
# max_len_a1 584 max_len_a2 231 max_len_a3 30 max_len_a4 27