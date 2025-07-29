import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=768):
        super(CNNEncoder, self).__init__()
        
        # Initial convolution and pooling
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Residual blocks with increasing channels and downsampling
        self.block1 = ResidualBlock3D(64, 128, downsample=True)
        # self.block2 = ResidualBlock3D(128, 256, downsample=True)
        # self.block3 = ResidualBlock3D(256, 512, downsample=True)
        
        # Final expansion to target channel size
        # self.conv_final = nn.Conv3d(512, out_channels, kernel_size=1)
        self.conv_final = nn.Conv3d(128, out_channels, kernel_size=1)
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Custom weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # He initialization for ReLU networks
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Small positive bias
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Special initialization for final layer
        nn.init.xavier_uniform_(self.conv_final.weight, gain=math.sqrt(2))
        nn.init.constant_(self.conv_final.bias, 0)
        
    def forward(self, x):
        # Input shape: (batch, channels, depth, height, width)
        # Assuming input is (batch, 1, 32, 256, 256)
        
        x = F.relu(self.bn1(self.conv1(x)))  # -> (batch, 64, 16, 128, 128)
        x = self.pool1(x)                    # -> (batch, 64, 8, 64, 64)
        
        x = self.block1(x)                   # -> (batch, 128, 4, 32, 32)
        # x = self.block2(x)                   # -> (batch, 256, 2, 16, 16)
        # x = self.block3(x)                   # -> (batch, 512, 1, 8, 8)
        
        # Use adaptive pooling instead of fixed interpolation
        x = F.adaptive_avg_pool3d(x, (4, 4, 4))  # More stable than interpolate
        x = self.conv_final(x)               # -> (batch, 768, 4, 4, 4)
        
        return x.permute(0, 2, 3, 4, 1)      # (batch, 4, 4, 4, 768)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock3D, self).__init__()
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 
                              kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        # Initialize residual block
        self._init_block()
        
    def _init_block(self):
        """Initialize residual block weights"""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        
        if isinstance(self.shortcut, nn.Sequential):
            nn.init.kaiming_normal_(self.shortcut[0].weight, mode='fan_out')
            nn.init.constant_(self.shortcut[1].weight, 1)
            nn.init.constant_(self.shortcut[1].bias, 0)
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        x += residual
        return F.relu(x)

if __name__ == "__main__":
    model = VisionEncoder3D(in_channels=1, out_channels=768)
    
    dummy_input = torch.randn(1, 1, 32, 256, 256)
    
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Should be (1, 4, 4, 4, 768)