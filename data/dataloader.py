import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json 
import pickle
import sys 
from transformers import AutoTokenizer
class TracDataset(Dataset):
    def __init__(self,model_type, mode,patient_records,root_dir, transform=None):
        self.tokenizer= AutoTokenizer.from_pretrained(model_type)
        self.mode = mode
        self.patient_records = patient_records
        self.root_dir = root_dir
        self.slice_records=[]
        for record in patient_records:
            path=os.path.join(root_dir,"data",record)
            with open(path,"r") as f:
                data=json.load(f)
                self.slice_records.extend(data)
        self.max_seqs_len=100
        self.prefix_len=300
     
    def __len__(self):
        return len(self.slice_records)
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
        slide_record=self.slice_records[idx]
        print("slide_record",slide_record,type(slide_record))
        print("pid",slide_record["Patient ID"],"idx",idx)
        chunk_idx=slide_record["chunk_idx"]
        patient_id=slide_record["Patient ID"]
        image_path = os.path.join(self.root_dir, "image", patient_id,f'{chunk_idx}.pkl')
        with open(image_path, "rb") as f:
            image = pickle.load(f)
        sys.stdout.flush()
        
        q1,a1,q2,a2,q3,a3,q4,a4=slide_record["Q1"],slide_record["A1"],slide_record["Q2"],slide_record["A2"],slide_record["Q3"],slide_record["A3"],slide_record["Q4"],slide_record["A4"]
        index=slide_record["slide_in_chunk_th"]
        q1,a1,idx1=self.process_input_text(q1,a1)
        q2,a2,idx2=self.process_input_text(q2,a2)
        q3,a3,idx3=self.process_input_text(q3,a3)
        q4,a4,idx4=self.process_input_text(q4,a4)
        assert idx1==idx2==idx3==idx4 
        return{
        "image":image,
        "q1":q1,
        "a1":a1,
        "q2":q2,
        "a2":a2,
        "q3":q3,
        "a3":a3,
        "q4":q4,
        "a4":a4,
        "index":idx1

        } 
    # image,q1,a1,q2,a2,q3,a3,q4,a4,index
if __name__=="__main__":
    import os
    import random
    from sklearn.model_selection import train_test_split
    train_val_dir="/home/ubuntu/repo/TracGPT-R3D/clean_data_3d/50_slices/3393c76c-3917-4791-83e7-e32936431012/train"
    patient_records=os.listdir(os.path.join(train_val_dir,"data"))
    patient_records = sorted(patient_records) 
    train_records, val_records = train_test_split(patient_records, test_size=0.2, random_state=42)
    train_set=TracDataset("gpt2","train",train_records,train_val_dir)
    image,q1,a1,q2,a2,q3,a3,q4,a4,index=train_set[0]
    print("image",image.shape, "q1",q1.shape,"a1",a1.shape,"q2",q2.shape,"a2",a2.shape,"q3",q3.shape,"a3",a3.shape,"q4",q4.shape,"a4",a4.shape,"index",index)

# max_len_q1 31 max_len_q2 67 max_len_q3 63 max_len_q4 63
# max_len_a1 584 max_len_a2 231 max_len_a3 30 max_len_a4 27