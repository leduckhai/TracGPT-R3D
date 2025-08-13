from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(num_classes=2)  # 1 class + background
model.train()

images = [torch.rand(3, 512, 512)]
targets = [{
    "boxes": torch.tensor([[50., 30., 200., 180.]]),
    "labels": torch.tensor([1]),
}]

losses = model(images, targets)
print(losses)  

total_loss = sum(losses.values())
