# models/backbone.py
"""
Image backbone: EfficientNet-B4 with multi-scale feature extraction.
Extracts features at stride 8 and stride 16 for the BEV transform.
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B4 backbone for multi-camera feature extraction.
    
    Extracts features at two scales:
    - stride 8:  160 channels (for fine-grained depth prediction)
    - stride 16: 272 channels (for context)
    
    Features are fused into a single output at stride 8.
    """

    def __init__(self, out_channels=160, pretrained=True):
        super().__init__()
        self.out_channels = out_channels

        # Load pretrained EfficientNet-B4
        if pretrained:
            self.backbone = EfficientNet.from_pretrained("efficientnet-b4")
        else:
            self.backbone = EfficientNet.from_name("efficientnet-b4")

        # EfficientNet-B4 actual block output channels:
        # reduction_1: stride 2,  24ch
        # reduction_2: stride 4,  32ch
        # reduction_3: stride 8,  56ch
        # reduction_4: stride 16, 160ch
        # reduction_5: stride 32, 448ch

        # We'll extract from the EfficientNet endpoints
        # and build a small FPN-like neck
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Lateral connections (using actual channel counts)
        self.lateral_s16 = nn.Conv2d(448, out_channels, 1)
        self.lateral_s8 = nn.Conv2d(160, out_channels, 1)

        # Smooth after fusion
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B*N, 3, H, W) where N = number of cameras
        Returns:
            features: (B*N, out_channels, H/8, W/8)
        """
        # Extract features using EfficientNet's internal method
        endpoints = self._extract_endpoints(x)

        # FPN-style fusion
        p_s16 = self.lateral_s16(endpoints["stride_32"])  # Actually stride 16 in our naming
        p_s8 = self.lateral_s8(endpoints["stride_16"])

        # Top-down fusion
        p_s8 = p_s8 + self.up(p_s16)
        out = self.smooth(p_s8)

        return out

    def _extract_endpoints(self, x):
        """Extract multi-scale features from EfficientNet."""
        endpoints = {}

        # Use EfficientNet's extract_endpoints method
        feats = self.backbone.extract_endpoints(x)

        # Map to our naming convention
        # EfficientNet-B4 reduction levels:
        # reduction_3: stride 8   → 56 channels
        # reduction_4: stride 16  → 160 channels
        # reduction_5: stride 32  → 448 channels
        endpoints["stride_16"] = feats["reduction_4"]  # 160ch
        endpoints["stride_32"] = feats["reduction_5"]  # 448ch

        return endpoints


class DepthNet(nn.Module):
    """
    Predicts a depth distribution for each pixel in the image features.
    
    For each pixel, outputs D probabilities (softmax over depth bins).
    This is the "Lift" step in Lift-Splat-Shoot.
    """

    def __init__(self, in_channels=160, mid_channels=256, depth_channels=112,
                 use_dcn=False):
        super().__init__()
        self.depth_channels = depth_channels

        # Context net: predicts both depth and context features
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Separate heads for depth and context
        self.depth_head = nn.Conv2d(mid_channels, depth_channels, 1)
        self.context_head = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, features):
        """
        Args:
            features: (B*N, C, H, W) image features
        Returns:
            depth: (B*N, D, H, W) depth distribution (softmax)
            context: (B*N, C, H, W) context features
        """
        x = self.depth_net(features)

        depth = self.depth_head(x)
        depth = depth.softmax(dim=1)  # Probability over depth bins

        context = self.context_head(x)

        return depth, context
