import torch 
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
    