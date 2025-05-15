import torch
import torch.nn as nn

# Implementation of ASFF (Adaptive Spatial Feature Fusion) and BiFPN modules for Ultralytics YOLO

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network Block with learnable weights.
    Processes features from multiple scales and performs weighted bidirectional feature fusion.
    """
    
    def __init__(self, p5_channels, p4_channels, p3_channels):
        super().__init__()
        self.p5_channels = p5_channels
        self.p4_channels = p4_channels
        self.p3_channels = p3_channels
        
        # Epsilon for weight stability
        self.epsilon = 1e-4
        
        # Weights for feature fusion (learnable)
        # TD: Top-down pathway, BU: Bottom-up pathway
        self.p5_td_w = nn.Parameter(torch.ones(2))
        self.p4_td_w = nn.Parameter(torch.ones(3))
        self.p3_td_w = nn.Parameter(torch.ones(2))
        self.p4_bu_w = nn.Parameter(torch.ones(3))
        self.p5_bu_w = nn.Parameter(torch.ones(2))
        
        # Lateral connections and fusion convs
        # Top-down pathway
        self.p5_td_conv = nn.Conv2d(p5_channels, p4_channels, kernel_size=1, bias=False)
        self.p4_td_conv = nn.Conv2d(p4_channels, p3_channels, kernel_size=1, bias=False)
        
        # Top-down fusion convs
        self.p4_td_fusion = nn.Conv2d(p4_channels, p4_channels, kernel_size=3, padding=1, bias=False)
        self.p3_td_fusion = nn.Conv2d(p3_channels, p3_channels, kernel_size=3, padding=1, bias=False)
        
        # Bottom-up pathway
        self.p3_bu_conv = nn.Conv2d(p3_channels, p4_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.p4_bu_conv = nn.Conv2d(p4_channels, p5_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Bottom-up fusion convs
        self.p4_bu_fusion = nn.Conv2d(p4_channels, p4_channels, kernel_size=3, padding=1, bias=False)
        self.p5_bu_fusion = nn.Conv2d(p5_channels, p5_channels, kernel_size=3, padding=1, bias=False)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        # Unpack input features from different levels
        p5_in, p4_in, p3_in = inputs
        
        # Calculate weights using softmax for normalization
        p5_td_w = self.relu(self.p5_td_w)
        p5_td_w = p5_td_w / (torch.sum(p5_td_w) + self.epsilon)
        
        p4_td_w = self.relu(self.p4_td_w)
        p4_td_w = p4_td_w / (torch.sum(p4_td_w) + self.epsilon)
        
        p3_td_w = self.relu(self.p3_td_w)
        p3_td_w = p3_td_w / (torch.sum(p3_td_w) + self.epsilon)
        
        p4_bu_w = self.relu(self.p4_bu_w)
        p4_bu_w = p4_bu_w / (torch.sum(p4_bu_w) + self.epsilon)
        
        p5_bu_w = self.relu(self.p5_bu_w)
        p5_bu_w = p5_bu_w / (torch.sum(p5_bu_w) + self.epsilon)
        
        # Top-down pathway (from high-level to low-level)
        # P5 -> P4 (top-down)
        p5_td = self.p5_td_conv(p5_in)
        p5_td = F.interpolate(p5_td, scale_factor=2, mode='nearest')
        p4_td = p5_td_w[0] * p4_in + p5_td_w[1] * p5_td
        p4_td = self.relu(self.p4_td_fusion(p4_td))
        
        # P4 -> P3 (top-down)
        p4_td_to_p3 = self.p4_td_conv(p4_td)
        p4_td_to_p3 = F.interpolate(p4_td_to_p3, scale_factor=2, mode='nearest')
        p3_td = p4_td_w[0] * p3_in + p4_td_w[1] * p4_td_to_p3
        p3_out = self.relu(self.p3_td_fusion(p3_td))
        
        # Bottom-up pathway (from low-level to high-level)
        # P3 -> P4 (bottom-up)
        p3_bu = self.p3_bu_conv(p3_out)
        p4_bu = p4_bu_w[0] * p4_in + p4_bu_w[1] * p4_td + p4_bu_w[2] * p3_bu
        p4_out = self.relu(self.p4_bu_fusion(p4_bu))
        
        # P4 -> P5 (bottom-up)
        p4_bu_to_p5 = self.p4_bu_conv(p4_out)
        p5_bu = p5_bu_w[0] * p5_in + p5_bu_w[1] * p4_bu_to_p5
        p5_out = self.relu(self.p5_bu_fusion(p5_bu))
        
        return p5_out, p4_out, p3_out


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion Block for enhancing multi-scale feature extraction.
    Dynamically adjusts the pyramid feature contribution of each scale.
    """
    
    def __init__(self, level0_channels, level1_channels, level2_channels):
        super().__init__()
        self.level0_channels = level0_channels  # P3 channels
        self.level1_channels = level1_channels  # P4 channels
        self.level2_channels = level2_channels  # P5 channels
        
        # Inter-level feature fusion
        # Compress and expand channels for aligned feature fusion
        self.level0_to_level0 = nn.Identity()
        self.level1_to_level0 = nn.Sequential(
            nn.Conv2d(level1_channels, level0_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(level0_channels),
            nn.ReLU(inplace=True)
        )
        self.level2_to_level0 = nn.Sequential(
            nn.Conv2d(level2_channels, level0_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(level0_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention weights for adaptive fusion
        # Generate per-pixel attention maps for each level
        self.spatial_weights = nn.Conv2d(level0_channels * 3, 3, kernel_size=1, bias=False)
        self.spatial_attn = nn.Sigmoid()
        
        # Final fusion conv
        self.fusion_conv = nn.Conv2d(level0_channels, level0_channels, kernel_size=3, padding=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(level0_channels)
        self.fusion_act = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        # Unpack multi-scale features from different levels
        level0, level1, level2 = inputs  # P3, P4, P5
        
        # Spatial size adaptation (resize to level0 size, which is P3)
        level0_size = level0.shape[-2:]
        
        # Resize all features to the P3 size
        level1_resized = F.interpolate(level1, size=level0_size, mode='bilinear', align_corners=False)
        level2_resized = F.interpolate(level2, size=level0_size, mode='bilinear', align_corners=False)
        
        # Channel adaptation
        level0_adapted = self.level0_to_level0(level0)
        level1_adapted = self.level1_to_level0(level1_resized)
        level2_adapted = self.level2_to_level0(level2_resized)
        
        # Concatenate features for spatial attention
        concat_features = torch.cat([level0_adapted, level1_adapted, level2_adapted], dim=1)
        
        # Generate spatial attention weights
        attn_weights = self.spatial_weights(concat_features)
        attn_weights = self.spatial_attn(attn_weights)
        
        # Split attention weights for each level
        level0_weight, level1_weight, level2_weight = torch.chunk(attn_weights, chunks=3, dim=1)
        
        # Apply attention weights for adaptive fusion
        fused_features = (level0_adapted * level0_weight + 
                         level1_adapted * level1_weight + 
                         level2_adapted * level2_weight)
        
        # Final fusion
        output = self.fusion_act(self.fusion_bn(self.fusion_conv(fused_features)))
        
        return output
