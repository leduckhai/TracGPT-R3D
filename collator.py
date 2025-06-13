pythonsample = {
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
            'answer': "[2.5, 1.2, 0.8, 4.2, 1.8, 1.5, 0.0, 0.0, 1.57]",  # Serialized bbox
            'answer_type': 'bbox_3d',
            'bbox_3d': [2.5, 1.2, 0.8, 4.2, 1.8, 1.5, 0.0, 0.0, 1.57]  # Ground truth
        },
        {
            'question': "Where is the bounding box of the person?",
            'answer': "<bbox>120,50,200,180</bbox>",  # 2D bbox in special tokens
            'answer_type': 'bbox_2d',
            'bbox_2d': [120, 50, 200, 180]
        }
    ]
}

class BboxAwareCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add special bbox tokens if not already in tokenizer
        special_tokens = ["<bbox>", "</bbox>", "<x>", "</x>", "<y>", "</y>", "<z>", "</z>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    def format_bbox_answer(self, bbox, answer_type):
        """Convert bbox coordinates to formatted string"""
        if answer_type == 'bbox_2d':
            return f"<bbox>{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}</bbox>"
        elif answer_type == 'bbox_3d':
            # Format as structured text or JSON-like string
            return f"[{','.join([f'{x:.2f}' for x in bbox])}]"
        return None
    
    def __call__(self, batch):
        images = []
        input_ids = []
        attention_masks = []
        labels = []
        bbox_3d_gt = []  # Ground truth for auxiliary loss
        bbox_3d_mask = []
        answer_types = []
        
        for sample in batch:
            images.append(sample['image'])
            answer_types.append(sample['answer_type'])
            
            # Create input text (question only for generation)
            question_text = f"Question: {sample['question']} Answer:"
            
            # Create full text (question + answer for training)
            if sample['answer_type'] in ['bbox_2d', 'bbox_3d'] and sample.get('bbox_3d') is not None:
                # Use formatted bbox answer
                formatted_answer = self.format_bbox_answer(
                    sample['bbox_3d'] if sample['answer_type'] == 'bbox_3d' else sample.get('bbox_2d'),
                    sample['answer_type']
                )
                full_text = f"Question: {sample['question']} Answer: {formatted_answer}"
                bbox_3d_gt.append(sample.get('bbox_3d', [0]*9))
                bbox_3d_mask.append(sample['answer_type'] == 'bbox_3d')
            else:
                full_text = f"Question: {sample['question']} Answer: {sample['answer']}"
                bbox_3d_gt.append([0]*9)
                bbox_3d_mask.append(False)
            
            # Tokenize
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
        
        return {
            'images': torch.stack(images),
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': torch.stack(labels),
            'bbox_3d_gt': torch.tensor(bbox_3d_gt, dtype=torch.float32),
            'bbox_3d_mask': torch.tensor(bbox_3d_mask),
            'answer_types': answer_types
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