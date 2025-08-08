
import torch 

def compute_loss_bbox( model, inputs, return_outputs=False, **kwargs):
        """Compute the custom loss for TracVisionModel"""
        images = inputs.get("images")
        bbox_gts = inputs.get("center_bbox_gts")
        bbox_masks = inputs.get("bbox_masks")
        labels = inputs.get("labels")
        
        if bbox_gts is None:
            print("⚠️ bbox_gts is None")
            return 
        if bbox_masks is None:
            print("⚠️ bbox_masks is None")
            return
            
        outputs = model(
            images=images,
            bbox_gts=bbox_gts,
            bbox_masks=bbox_masks,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=images.device, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss
def compute_loss_raw_vision( model, inputs, return_outputs=False, **kwargs):
    image=inputs.get("images").to(model.device)
    bbox_criteria=inputs.get("bbox_criteria")
    status_criteria=inputs.get("status_criteria")
    outputs=model(image, bbox_criteria, status_criteria)
    loss=outputs.loss
    return (loss, outputs) if return_outputs else loss

def load_loss_compute(mode):
    if mode=="raw_vision":
        return compute_loss_raw_vision
    elif mode=="bbox":
        return compute_loss_bbox
    