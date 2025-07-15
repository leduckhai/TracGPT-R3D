from transformers import AutoTokenizer, AutoModel
from typing import Optional
import torch
import numpy as np

device = torch.device("cuda") # or cpu

tokenizer = AutoTokenizer.from_pretrained(
    "GoodBaiBai88/M3D-CLIP",
    model_max_length=512,
    padding_side="right",
    use_fast=False
)
model = AutoModel.from_pretrained(
    "GoodBaiBai88/M3D-CLIP",
    trust_remote_code=True
)
model = model.to(device=device)

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, considering resize and other methods.
# 2. The image needs to be normalized to 0-1, considering Min-Max Normalization.
# 3. The image format needs to be converted to .npy 
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.
    
image_path = ""
input_txt = "test random"

text_tensor = tokenizer(input_txt, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
input_id = text_tensor["input_ids"].to(device=device)
attention_mask = text_tensor["attention_mask"].to(device=device)
# image = np.load(image_path).to(device=device)
image=torch.randn(1,1,32, 256, 256).to(device=device)  # Dummy image tensor

with torch.inference_mode():
    image_features = model.encode_image(image)[:, 0]
    # text_features = model.encode_text(input_id, attention_mask)[:, 0]
    print("Image features shape:", image_features.shape)