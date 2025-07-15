import torch 
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List
import numpy as np
import torch.nn.functional as F
from model.bbox3d.helper import corners_to_center

class BboxAwareCollator:
    def __init__(self, tokenizer, max_length=512, max_bbox_length=9, num_vision_token=256,token_name="<image>",end_token="<end>"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_bbox_length = max_bbox_length
        self.end_token = end_token
        self.image_tk_name=token_name
        self.image_tk = f"<image_context> {' '.join([token_name] * num_vision_token)} <image_context>"
    def pad_bboxes_to_fixed_size(self, bboxes: List[List[float]], target_size: int) -> List[List[float]]:
        """Pad or truncate bboxes to fixed size"""
        if len(bboxes) >= target_size:
            return bboxes[:target_size] 
        else:
            padding_bbox = [0.0] * len(bboxes[0]) if bboxes else [0.0] * 6  
            padded = bboxes + [padding_bbox] * (target_size - len(bboxes))
            return padded
        
    def format_bbox_answer(self, bboxes, answer_type):
        """Convert bbox coordinates to formatted string"""
        if answer_type == 'bbox_3d':
            output = []
            for bbox in bboxes:
                # Assuming bbox format: [x_min, x_max, y_min, y_max, z_min, z_max]
                bbox_str = (f"<bbox><x>{bbox[0]:.3f},{bbox[1]:.3f}</x>"
                           f"<y>{bbox[2]:.3f},{bbox[3]:.3f}</y>"
                           f"<z>{bbox[4]:.3f},{bbox[5]:.3f}</z></bbox>")
                output.append(bbox_str)
            return "".join(output)
        else:
            return str(bboxes)
        
    def create_bbox_attention_mask(self, length: int, max_len: int) -> List[List[bool]]:
        """Create attention mask for padded bboxes"""
        length=min(length,max_len)
        mask = [True] * length + [False] * (max_len - length)
        return mask
    
    def __call__(self, batch):
        images = []
        input_ids = []
        attention_masks = []
        labels = []
        bbox_gts=[
        ]
        bbox_masks=[]
        answer_types = []
        questions=[]
        answers=[]
        for sample in batch:
            images.append(sample['image'])
            answer_types.append(sample['answer_type'])
            
            if sample['answer_type'] in ['bbox_3d'] and sample.get('bbox_3d') is not None:
                bbox_data = sample['bbox_3d'] 

                bbox_gt= self.pad_bboxes_to_fixed_size(bbox_data, self.max_bbox_length)  # Pad to fixed size of 2 bboxes
                
                bbox_mask=self.create_bbox_attention_mask(len(bbox_data), self.max_bbox_length)
                
                formatted_answer = self.format_bbox_answer(bbox_data, sample['answer_type'])
                
            else:
                formatted_answer = sample['answer']
                full_text = f"Question: {sample['question']} Answer: {formatted_answer}"
                bbox_gt=self.pad_bboxes_to_fixed_size([[0.0]*6], self.max_bbox_length)
                bbox_mask=self.create_bbox_attention_mask(0, self.max_bbox_length)
            
            questions.append(sample['question'])
            answers.append(formatted_answer)
            question_text = f"Question: {sample['question']} Answer:"
            full_text = f"{question_text} {self.image_tk} Answer: {formatted_answer} {self.end_token}"
            bbox_gts.append(torch.tensor(bbox_gt, dtype=torch.float32))
            bbox_masks.append(torch.tensor(bbox_mask, dtype=torch.bool))
            encoded = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded['input_ids'].squeeze(0))
            attention_masks.append(encoded['attention_mask'].squeeze(0))
            
            
            label = encoded['input_ids'].clone().squeeze(0)
            question_len = len(self.tokenizer(question_text, return_tensors="pt")["input_ids"][0])
            label[:question_len] = -100
            labels.append(label)
            # print("n vision token", (encoded['input_ids']==self.tokenizer.convert_tokens_to_ids(self.image_tk_name)).sum().item())
        
        position_ids = torch.arange(0, self.max_length).expand(len(batch), -1).long()
        
        bbox_gts = torch.stack(bbox_gts)
        bbox_gts = torch.clamp(bbox_gts, min=1e-3, max=1.0)  
        bbox_gt=corners_to_center(bbox_gts)  # Convert to center format if needed
        return {
            'images': torch.stack(images),
            'input_ids': torch.stack(input_ids),
            'attention_masks': torch.stack(attention_masks),
            'labels': torch.stack(labels),
            'bbox_gts': bbox_gts,
            'bbox_masks': torch.stack( bbox_masks),
            'position_ids': position_ids  ,
            'answer_types': answer_types,
            'questions':questions,
            'answers':answers   
        }
    

    
    def validate_coordinates(self, coords, answer_type, image_size=None):
        """Validate extracted coordinates"""
        if answer_type == 'bbox_2d' and len(coords) == 4:
            x, y, w, h = coords
            if image_size:
                img_w, img_h = image_size
                return 0 <= x <= img_w and 0 <= y <= img_h and w > 0 and h > 0
        elif answer_type == 'bbox_3d' and len(coords) == 9:
            # Basic validation for 3D coordinates
            return all(isinstance(c, (int, float)) for c in coords)
        return False


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tokenizer=AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    img_token_name="<im_patch>"
    tokenizer.add_tokens(img_token_name)
    img_id= tokenizer.convert_tokens_to_ids(img_token_name)
    print("img id",img_id)
    for batch in dl:
        images, input_ids, attention_mask, labels, bbox_gt, bbox_mask,position_ids = batch.values()
        print("images shape:", images.shape)
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("labels shape:", labels.shape)
        print("bbox_3d_mask shape:", bbox_mask.shape)
        print("position_ids shape:", position_ids.shape)
        print("input ids",input_ids)
        print("input id", (input_ids==img_id).sum().item())
        break