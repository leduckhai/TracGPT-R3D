import torch 
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List

class QA3DDataset(Dataset):
    def __init__(self, data=None):
        image_tensor = torch.randn(1,32, 256, 256)  # Simulated image tensor
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
                    'answer': "[[2.5, 1.2, 0.8, 4.2, 1.8, 2.3],[2.0, 1.2, 0.8, 4.2, 1.8, 1.5]]", 
                    'answer_type': 'bbox_3d',
                    'bbox_3d':[[2.5, 1.2, 0.8, 4.2, 1.8, 2.3],[2.0, 1.2, 0.8, 4.2, 1.8, 1.5]]
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
    def __init__(self, tokenizer, max_length=512, max_bbox_length=9):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_bbox_length = max_bbox_length

        special_tokens = ["<bbox>", "</bbox>", "<x>", "</x>", "<y>", "</y>", "<z>", "</z>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
      
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
        length=max(length,max_len)
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
            
            # Create input text (question only for generation)
            question_text = f"Question: {sample['question']} Answer:"
            
            # Create full text (question + answer for training)
            if sample['answer_type'] in ['bbox_3d'] and sample.get('bbox_3d') is not None:
                bbox_data = sample['bbox_3d'] 

                bbox_gt= self.pad_bboxes_to_fixed_size(bbox_data, self.max_bbox_length)  # Pad to fixed size of 2 bboxes
                
                bbox_mask=self.create_bbox_attention_mask(len(bbox_data), self.max_bbox_length)
                
                formatted_answer = self.format_bbox_answer(bbox_data, sample['answer_type'])
                full_text = f"Question: {sample['question']} Answer: {formatted_answer}"
                
            else:
                full_text = f"Question: {sample['question']} Answer: {sample['answer']}"
                bbox_gt=self.pad_bboxes_to_fixed_size([[0.0]*6], self.max_bbox_length)
                bbox_mask=self.create_bbox_attention_mask(0, self.max_bbox_length)
            bbox_gts.append(bbox_gt)
            bbox_masks.append(bbox_mask)
            encoded = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded['input_ids'].squeeze(0))
            attention_masks.append(encoded['attention_mask'].squeeze(0))
            
            # Create labels (mask question part, only train on answer)
            question_encoded = self.tokenizer(question_text, add_special_tokens=False)
            question_length = len(question_encoded['input_ids'])
            
            label = encoded['input_ids'].clone().squeeze(0)
            label[:question_length] = -100  # Ignore question tokens in loss
            labels.append(label)
        
        # Create position_ids with correct batch size
        position_ids = torch.arange(0, self.max_length).expand(len(batch), -1).long()
        
        return {
            'images': torch.stack(images),
            'input_ids': torch.stack(input_ids),
            'attention_masks': torch.stack(attention_masks),
            'labels': torch.stack(labels),
            'bbox_gts': torch.stack([torch.tensor(b) for b in bbox_gts]),
            'bbox_masks': torch.stack([torch.tensor(b) for b in bbox_masks]),
            'answer_types': answer_types,
            'position_ids': position_ids   
        }
    
class MultiModalBboxModel(nn.Module):
    def __init__(self, language_model, vision_encoder, hidden_size=4096):
        super().__init__()
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        
        # Auxiliary bbox regression head
        self.bbox_3d_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9)  # x,y,z,w,h,d,rx,ry,rz
        )
        
        # Coordinate embedding for better numerical understanding
        self.coord_embed = nn.Embedding(1000, hidden_size // 4)  # For discretized coords
        
    def forward(self, images, input_ids, attention_mask, labels=None, 
                bbox_3d_gt=None, bbox_3d_mask=None, answer_types=None):
        
        # Vision encoding
        vision_features = self.vision_encoder(images)
        
        # Main language modeling loss
        lm_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=labels
        )
        
        outputs = {
            'lm_loss': lm_outputs.loss,
            'logits': lm_outputs.logits,
            'generated_text': None  # Will be filled during inference
        }
        
        # Auxiliary bbox regression loss (helps with coordinate understanding)
        if bbox_3d_mask is not None and bbox_3d_mask.any():
            bbox_samples = bbox_3d_mask.bool()
            bbox_vision_features = vision_features[bbox_samples]
            
            # Get text features for bbox questions
            last_hidden = lm_outputs.hidden_states[-1][bbox_samples] if hasattr(lm_outputs, 'hidden_states') else None
            
            if last_hidden is not None:
                # Combine vision and text features
                combined_features = torch.cat([
                    bbox_vision_features.mean(dim=1),  # Pool vision features
                    last_hidden.mean(dim=1)  # Pool text features
                ], dim=-1)
                
                bbox_pred = self.bbox_3d_head(combined_features)
                bbox_targets = bbox_3d_gt[bbox_samples]
                
                bbox_loss = F.smooth_l1_loss(bbox_pred, bbox_targets)
                outputs['bbox_3d_loss'] = bbox_loss
                outputs['bbox_3d_pred'] = bbox_pred
        
        return outputs
    

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


def train_step(model, batch, optimizer, alpha=0.1):
    """Train with both language modeling and auxiliary bbox loss"""
    outputs = model(**batch)
    
    # Primary loss: language modeling
    total_loss = outputs['lm_loss']
    
    # Auxiliary loss: direct bbox regression (helps numerical understanding)
    if 'bbox_3d_loss' in outputs:
        total_loss += alpha * outputs['bbox_3d_loss']
    
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return {
        'total_loss': total_loss.item(),
        'lm_loss': outputs['lm_loss'].item(),
        'bbox_loss': outputs.get('bbox_3d_loss', torch.tensor(0.0)).item()
    }

def generate_answer(model, tokenizer, processor, image, question, answer_type):
    model.eval()
    
    # Prepare input
    prompt = f"Question: {question} Answer:"
    inputs = processor(image, prompt)
    
    with torch.no_grad():
        # Generate text
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract coordinates if bbox question
        if answer_type in ['bbox_2d', 'bbox_3d']:
            coords = processor.extract_coordinates_from_text(generated_text, answer_type)
            return {
                'text': generated_text,
                'coordinates': coords,
                'valid': processor.validate_coordinates(coords, answer_type)
            }
        
        return {'text': generated_text}

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    ds= QA3DDataset()
    print("Dataset length:", len(ds))
    tokenizer=AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    dl= torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=BboxAwareCollator(tokenizer=tokenizer))
    for batch in dl:
        images, input_ids, attention_mask, labels, bbox_gt, bbox_mask, answer_types,position_ids = batch.values()
        print("images shape:", images.shape)
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("labels shape:", labels.shape)
        print("bbox_3d_gt shape:", bbox_gt.shape, bbox_gt)
        print("bbox_3d_mask shape:", bbox_mask.shape)
        print("answer_types:", answer_types)
        print("position_ids shape:", position_ids.shape)