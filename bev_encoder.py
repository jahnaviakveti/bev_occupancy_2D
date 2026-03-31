# models/bev_encoder.py
"""
BEV Encoder & Occupancy Head.

The BEV encoder processes the BEV features with a ResNet-like network,
producing multi-scale features. The occupancy head predicts binary
occupancy at each scale for deep supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Basic residual block for BEV encoding."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class BEVEncoder(nn.Module):
    """
    Multi-scale BEV feature encoder.
    
    Takes BEV features and produces features at 3 scales:
    - scale_1: 1x resolution (200x200)
    - scale_2: 1/2 resolution (100x100)  
    - scale_3: 1/4 resolution (50x50)
    
    Then upsamples back for multi-scale fusion.
    """

    def __init__(self, in_channels=64, channels=(64, 128, 256)):
        super().__init__()
        c1, c2, c3 = channels

        # Scale 1: same resolution
        self.layer1 = nn.Sequential(
            ResBlock(in_channels, c1),
            ResBlock(c1, c1),
        )

        # Scale 2: downsample 2x
        self.layer2 = nn.Sequential(
            ResBlock(c1, c2, stride=2),
            ResBlock(c2, c2),
        )

        # Scale 3: downsample 2x more
        self.layer3 = nn.Sequential(
            ResBlock(c2, c3, stride=2),
            ResBlock(c3, c3),
        )

        # Upsample paths for FPN-style fusion
        self.up3 = nn.Sequential(
            nn.Conv2d(c3, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(c2, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.out_channels = [c1, c2, c3]

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) BEV features
        Returns:
            multi_scale: list of [(B, c1, H, W), (B, c2, H/2, W/2), (B, c3, H/4, W/4)]
            fused: (B, c1, H, W) final fused features at full resolution
        """
        # Encode
        f1 = self.layer1(x)   # (B, c1, H, W)
        f2 = self.layer2(f1)  # (B, c2, H/2, W/2)
        f3 = self.layer3(f2)  # (B, c3, H/4, W/4)

        # Decode with skip connections
        p2 = self.fuse2(f2 + self.up3(f3))    # (B, c2, H/2, W/2)
        p1 = self.fuse1(f1 + self.up2(p2))    # (B, c1, H, W)

        return [p1, p2, f3], p1


class OccupancyHead(nn.Module):
    """
    Multi-scale occupancy prediction head with deep supervision.
    
    Predicts occupancy probability at each BEV encoder scale.
    During training, all scales are supervised.
    During inference, only the finest scale is used.
    """

    def __init__(self, in_channels_list=(64, 128, 256), num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        # One prediction head per scale
        self.heads = nn.ModuleList()
        for c in in_channels_list:
            self.heads.append(
                nn.Sequential(
                    nn.Conv2d(c, c // 2, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c // 2, num_classes, 1),
                )
            )

    def forward(self, multi_scale_features, target_size=None):
        """
        Args:
            multi_scale_features: list of BEV features at different scales
            target_size: (H, W) to upsample predictions to (for loss computation)
        
        Returns:
            predictions: list of (B, num_classes, H_i, W_i) logits per scale
                         If target_size is given, all are upsampled to that size.
        """
        predictions = []
        for feat, head in zip(multi_scale_features, self.heads):
            pred = head(feat)
            if target_size is not None:
                pred = F.interpolate(
                    pred, size=target_size, mode="bilinear", align_corners=True
                )
            predictions.append(pred)

        return predictions
