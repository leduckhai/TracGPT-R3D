import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class TrainableFasterRCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, trainable_backbone_layers=3,device='cuda'):
        """
        Args:
            num_classes (int): number of output classes (including background)
            pretrained (bool): if True, uses pretrained weights
            trainable_backbone_layers (int): number of backbone layers to train (0-5)
        """
        super(TrainableFasterRCNN, self).__init__()
        
        # Load pretrained Faster R-CNN with ResNet-50 FPN
        self.model = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.device = device
    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image
        Returns:
            losses (dict[Tensor]) during training, or detections during inference
        """
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be provided")
            return self.model(images, targets)
        else:
            return self.model(images)

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = TrainableFasterRCNN(num_classes=2, pretrained=True, trainable_backbone_layers=3)
    model.train()  # set to training mode
    
    # Create example data
    images = [torch.rand(3, 512, 512)]
    targets = [{
        "boxes": torch.tensor([[50., 30., 200., 180.]]),
        # Shape: (num_objects, 4)
        "labels": torch.tensor([1]),
        # Shape: (num_objects,)
    }]
    
    # Forward pass
    losses = model(images, targets)
    print("Losses:", losses)
    
    # Compute total loss
    total_loss = sum(losses.values())
    print("Total loss:", total_loss.item())
    
    # Backward pass (in actual training)
    # total_loss.backward()