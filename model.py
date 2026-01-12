"""
RADNet: Image dehazing network based on retinal neuromorphic inspiration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinalBlock(nn.Module):
    """Retinal-inspired processing block"""
    
    def __init__(self, channels):
        super(RetinalBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class AdaptiveLayer(nn.Module):
    """Adaptive layer inspired by retinal adaptation"""
    
    def __init__(self, channels):
        super(AdaptiveLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RADNet(nn.Module):
    """
    RADNet: Retinal-inspired Adaptive Dehazing Network
    
    Architecture inspired by retinal processing mechanisms for image dehazing
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, num_blocks=6):
        super(RADNet, self).__init__()
        
        # Initial feature extraction
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Retinal-inspired processing blocks
        self.retinal_blocks = nn.ModuleList([
            RetinalBlock(base_channels) for _ in range(num_blocks)
        ])
        
        # Adaptive layers
        self.adaptive_layers = nn.ModuleList([
            AdaptiveLayer(base_channels) for _ in range(num_blocks)
        ])
        
        # Feature fusion
        self.fusion = nn.Conv2d(base_channels * num_blocks, base_channels, 1)
        
        # Output layer
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, x):
        # Initial feature extraction
        feat = self.relu(self.conv_in(x))
        
        # Multi-scale retinal processing
        features = []
        for retinal_block, adaptive_layer in zip(self.retinal_blocks, self.adaptive_layers):
            feat = retinal_block(feat)
            feat = adaptive_layer(feat)
            features.append(feat)
        
        # Feature fusion
        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)
        
        # Output
        out = self.conv_out(fused)
        
        # Residual connection with input
        out = out + x
        
        return out


def build_model(in_channels=3, out_channels=3, base_channels=64, num_blocks=6):
    """Build RADNet model"""
    return RADNet(in_channels, out_channels, base_channels, num_blocks)


if __name__ == '__main__':
    # Test model
    model = build_model()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
