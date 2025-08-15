import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import numpy as np
from typing import Optional, Tuple
import sys 
sys.path.append("/root/TracGPT-R3D")
from src.model.Encoder.rcnn_3d.sample_prosal import _sample_proposals
class MedicalNet3DEncoder(nn.Module):
    """
    MedicalNet 3D vision encoder for encoding medical images to features
    that can be concatenated with LLM embeddings.
    """
    
    def __init__(
        self, 
        model_name: str = "TencentMedicalNet/MedicalNet-Resnet10",
        output_dim: int = 768,  # Common LLM embedding dimension
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load pretrained MedicalNet
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the feature dimension from the backbone
        # MedicalNet-Resnet10 typically outputs 512 features
        backbone_dim = self.backbone.config.hidden_size if hasattr(self.backbone.config, 'hidden_size') else 512
        
        # Projection layer to match LLM embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Learnable position embeddings for spatial features
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, output_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape (batch_size, 1, 32, 256, 256)
            
        Returns:
            Encoded features of shape (batch_size, num_tokens, output_dim)
        """
        batch_size = x.size(0)
        
        # Forward through MedicalNet backbone
        features = self.backbone(x)
        
        # Handle different output formats
        if hasattr(features, 'last_hidden_state'):
            features = features.last_hidden_state
        elif hasattr(features, 'pooler_output'):
            features = features.pooler_output
        elif isinstance(features, tuple):
            features = features[0]
        
        # If features are 3D (batch, seq_len, hidden), use as is
        # If features are 2D (batch, hidden), add sequence dimension
        if features.dim() == 2:
            features = features.unsqueeze(1)  # (batch, 1, hidden)
        
        # Project to target dimension
        projected_features = self.projection(features)  # (batch, seq_len, output_dim)
        
        # Add position embeddings
        projected_features = projected_features + self.pos_embedding
        
        return projected_features

class MedicalVisionLLMProcessor:
    """
    Processor to handle the integration of MedicalNet features with LLM tokens
    """
    
    def __init__(self, vision_encoder: MedicalNet3DEncoder, tokenizer):
        self.vision_encoder = vision_encoder
        self.tokenizer = tokenizer
        
    def process_image_and_text(
        self, 
        image: torch.Tensor, 
        text: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process image and text for LLM input
        
        Args:
            image: 3D medical image tensor (1, 32, 256, 256)
            text: Input text string
            device: Device to run computation on
            
        Returns:
            Tuple of (combined_embeddings, attention_mask)
        """
        # Move to device
        image = image.to(device)
        self.vision_encoder = self.vision_encoder.to(device)
        
        # Encode image
        with torch.no_grad():
            image_features = self.vision_encoder(image.unsqueeze(0))  # Add batch dim
        
        # Tokenize text
        text_tokens = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Get text embeddings (assuming we have access to LLM embeddings)
        text_input_ids = text_tokens["input_ids"].to(device)
        text_attention_mask = text_tokens["attention_mask"].to(device)
        
        # Create attention mask for image features
        image_attention_mask = torch.ones(
            image_features.size(0), 
            image_features.size(1),
            dtype=torch.long,
            device=device
        )
        
        # Concatenate attention masks
        combined_attention_mask = torch.cat([
            image_attention_mask,
            text_attention_mask
        ], dim=1)
        
        return image_features, text_input_ids, combined_attention_mask

# Example usage and testing
if __name__ == "__main__":
    """
    Example of how to use the MedicalNet encoder with mock data
    """
    # Initialize encoder
    encoder = MedicalNet3DEncoder(
        model_name="TencentMedicalNet/MedicalNet-Resnet10",
        output_dim=768,  # Match your LLM embedding dimension
        freeze_backbone=True
    )
    
    # Create mock 3D medical image (batch_size=1, channels=1, depth=32, height=256, width=256)
    mock_image = torch.randn(1, 1, 32, 256, 256)
    
    # Encode image
    with torch.no_grad():
        encoded_features = encoder(mock_image)
    
    print(f"Input shape: {mock_image.shape}")
    print(f"Encoded features shape: {encoded_features.shape}")
    print(f"Feature dimension: {encoded_features.size(-1)}")
    