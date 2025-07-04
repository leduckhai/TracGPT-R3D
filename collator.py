import torch 
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List
import numpy as np
import torch.nn.functional as F

class QA3DDataset(Dataset):
    def __init__(self, data=None):
        image_tensor = torch.randn(1,32, 256, 256)  
        sample = {
            'image': image_tensor,
            'questions': [
                {
                    'question': "What color is the car?",
                    'answer': "Red",
                    'answer_type': 'text',
                    'bbox_3d': None
                },
                {
                    'question': "What are the 3D coordinates of the car?",
                    'answer': "[[0.3, 0.4, 0.8, 0.1, 0.6, 0.2],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]]", 
                    'answer_type': 'bbox_3d',
                    'bbox_3d':[[0.2, 0.4, 0.8, 0.1, 0.6, 0.2],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]]
                },
                 {
                    'question': "What color is the car?",
                    'answer': "Red",
                    'answer_type': 'text',
                    'bbox_3d': None
                },
                {
                    'question': "What are the 3D coordinates of the car?",
                    'answer': "[[0.3, 0.4, 0.8, 0.1, 0.6, 0.2],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]]", 
                    'answer_type': 'bbox_3d',
                    'bbox_3d':[[0.1, 0.4, 0.8, 0.1, 0.6, 0.2],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]]
                },
                 {
                    'question': "What color is the bike?",
                    'answer': "Red",
                    'answer_type': 'text',
                    'bbox_3d': None
                },
                {
                    'question': "What are the 3D coordinates of the bike?",
                    'answer': "[[0.3, 0.4, 0.8, 0.1, 0.6, 0.2],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]", 
                    'answer_type': 'bbox_3d',
                    'bbox_3d':[[0.3, 0.4, 0.8, 0.1, 0.6, 0.2],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]]
                },
                 {
                    'question': "What color is the board?",
                    'answer': "Red",
                    'answer_type': 'text',
                    'bbox_3d': None
                },
                {
                    'question': "What are the 3D coordinates of the baord?",
                    'answer': "[[0.3, 0.4, 0.8, 0.1, 0.6, 0.2],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]", 
                    'answer_type': 'bbox_3d',
                    'bbox_3d':[[0.5, 0.4, 0.8, 0.1, 0.6, 0.2],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3],[0.4, 0.02, 0.1, 0.1, 0.1, 0.3]]
                }

              
            ]
        }

        flattened_data = []
        for q in sample['questions']:
            flattened_data.append({
                'image': sample['image'],
                'question': q['question'],
                'answer': q['answer'],
                'answer_type': q['answer_type'],
                'bbox_2d': q.get('bbox_2d'),
                'bbox_3d': q.get('bbox_3d')
            })

        self.samples = flattened_data
        print("length of samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item 

class BboxAwareCollator:
    def __init__(self, tokenizer, max_length=512, max_bbox_length=9, num_vision_token=256,token_name="<image>"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_bbox_length = max_bbox_length
        # self.image_tk=f"<image_context>  {token_name*num_vision_token} <image_context>"
        self.image_tk = f"<image_context> {' '.join([token_name] * num_vision_token)} <image_context>"

    # Convert to 1D tensor
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
            
            question_text = f"Question: {sample['question']} Answer:"
            full_text = f"{question_text} {self.image_tk} Answer: {formatted_answer}"
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
        
        position_ids = torch.arange(0, self.max_length).expand(len(batch), -1).long()
        
        return {
            'images': torch.stack(images),
            'input_ids': torch.stack(input_ids),
            'attention_masks': torch.stack(attention_masks),
            'labels': torch.stack(labels),
            'bbox_gts': torch.stack([torch.tensor(b) for b in bbox_gts]),
            'bbox_masks': torch.stack([torch.tensor(b) for b in bbox_masks]),
            'position_ids': position_ids   
        }
    
class BboxPostProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def extract_coordinates_from_text(self, generated_text, answer_type):
        """Extract coordinates from generated text"""
        if answer_type == 'bbox_2d':
            # Extract from <bbox>x,y,w,h</bbox> format
            import re
            pattern = r'<bbox>([\d,\.]+)</bbox>'
            match = re.search(pattern, generated_text)
            if match:
                coords = [float(x.strip()) for x in match.group(1).split(',')]
                return coords
        
        elif answer_type == 'bbox_3d':
            # Extract from [x,y,z,w,h,d,rx,ry,rz] format
            import re
            pattern = r'\[([\d,\.\-\s]+)\]'
            match = re.search(pattern, generated_text)
            if match:
                coords = [float(x.strip()) for x in match.group(1).split(',')]
                return coords
        
        return None
    
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
    ds= QA3DDataset()
    print("Dataset length:", len(ds))
    tokenizer=AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    dl= torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=BboxAwareCollator(tokenizer=tokenizer))
    for batch in dl:
        images, input_ids, attention_mask, labels, bbox_gt, bbox_mask,position_ids = batch.values()
        print("images shape:", images.shape)
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("labels shape:", labels.shape)
        print("bbox_3d_gt shape:", bbox_gt.shape, bbox_gt)
        print("bbox_3d_mask shape:", bbox_mask.shape)
        print("position_ids shape:", position_ids.shape)