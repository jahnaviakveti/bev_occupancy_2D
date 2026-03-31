# models/bevocc.py
"""
BEV-Occ: Full model assembling backbone, BEV transform, temporal fusion,
encoder, and occupancy head.
"""

import torch
import torch.nn as nn

from .backbone import EfficientNetBackbone, DepthNet
from .bev_transform import LiftSplat
from .temporal_fusion import TemporalFusion
from .bev_encoder import BEVEncoder, OccupancyHead


class BEVOcc(nn.Module):
    """
    Bird's-Eye-View Occupancy prediction model.
    
    Pipeline:
    1. Extract multi-camera image features (EfficientNet-B4)
    2. Predict depth distribution + context features (DepthNet)
    3. Lift-Splat to create BEV features
    4. Temporal fusion with previous frame (ConvGRU)
    5. BEV encoder (multi-scale ResNet)
    6. Occupancy head (multi-scale prediction)
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]

        self.feat_dim = model_cfg["image_feat_dim"]
        self.use_temporal = model_cfg["use_temporal"]

        # 1. Image backbone
        self.backbone = EfficientNetBackbone(
            out_channels=self.feat_dim,
            pretrained=True,
        )

        # 2. Depth + Context prediction
        self.depth_net = DepthNet(
            in_channels=self.feat_dim,
            mid_channels=256,
            depth_channels=model_cfg["depth_channels"],
        )

        # 3. Lift-Splat BEV transform
        self.lift_splat = LiftSplat(
            feat_dim=self.feat_dim,
            depth_channels=model_cfg["depth_channels"],
            x_bound=tuple(model_cfg["bev_x_bound"]),
            y_bound=tuple(model_cfg["bev_y_bound"]),
            z_bound=tuple(model_cfg["bev_z_bound"]),
            depth_min=model_cfg["depth_min"],
            depth_max=model_cfg["depth_max"],
        )

        # 4. Temporal fusion
        if self.use_temporal:
            self.temporal = TemporalFusion(
                bev_dim=self.feat_dim,
                hidden_dim=model_cfg["temporal_hidden_dim"],
                x_bound=tuple(model_cfg["bev_x_bound"]),
                y_bound=tuple(model_cfg["bev_y_bound"]),
            )

        # 5. BEV encoder
        bev_in = model_cfg["temporal_hidden_dim"] if self.use_temporal else self.feat_dim
        self.bev_encoder = BEVEncoder(
            in_channels=bev_in,
            channels=tuple(model_cfg["bev_encoder_channels"]),
        )

        # 6. Occupancy head
        self.occ_head = OccupancyHead(
            in_channels_list=tuple(model_cfg["bev_encoder_channels"]),
            num_classes=model_cfg["occupancy_classes"],
        )

        # Grid size for reference
        self.nx = int((model_cfg["bev_x_bound"][1] - model_cfg["bev_x_bound"][0])
                      / model_cfg["bev_x_bound"][2])
        self.ny = int((model_cfg["bev_y_bound"][1] - model_cfg["bev_y_bound"][0])
                      / model_cfg["bev_y_bound"][2])

    def extract_img_features(self, images):
        """
        Extract features from multi-camera images.
        
        Args:
            images: (B, N, 3, H, W) multi-camera images
        Returns:
            features: (B, N, C, H', W') image features
            depth: (B, N, D, H', W') depth distributions
            context: (B, N, C, H', W') context features
        """
        B, N, C_img, H, W = images.shape

        # Flatten batch and camera dims
        imgs_flat = images.view(B * N, C_img, H, W)

        # Backbone
        feats = self.backbone(imgs_flat)  # (B*N, C, H', W')
        _, C, fH, fW = feats.shape

        # Depth + Context
        depth, context = self.depth_net(feats)

        # Reshape back
        feats = feats.view(B, N, C, fH, fW)
        depth = depth.view(B, N, -1, fH, fW)
        context = context.view(B, N, C, fH, fW)

        return feats, depth, context

    def forward(self, batch, prev_hidden=None):
        """
        Full forward pass.
        
        Args:
            batch: dict with keys:
                - images: (B, N, 3, H, W) multi-camera images
                - intrinsics: (B, N, 3, 3) camera intrinsics
                - extrinsics: (B, N, 4, 4) camera-to-ego transforms
                - ego_motion: (B, 4, 4) prev→curr ego transform (optional)
            prev_hidden: (B, C, nx, ny) previous temporal hidden state
        
        Returns:
            output: dict with keys:
                - predictions: list of occupancy logits per scale
                - hidden: current hidden state for temporal fusion
                - depth: predicted depth distributions
        """
        images = batch["images"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        B, N, _, img_h, img_w = images.shape

        # 1-2. Extract image features + depth
        feats, depth_dist, context = self.extract_img_features(images)

        # 3. Lift-Splat → BEV
        bev = self.lift_splat(
            feats, depth_dist, context,
            intrinsics, extrinsics,
            img_h, img_w,
        )  # (B, C, nx, ny)

        # 4. Temporal fusion
        if self.use_temporal:
            ego_motion = batch.get("ego_motion", None)
            hidden = self.temporal(bev, prev_hidden, ego_motion)
        else:
            hidden = bev

        # 5. BEV encoder
        multi_scale, fused = self.bev_encoder(hidden)

        # 6. Occupancy prediction (all scales upsampled to full res)
        predictions = self.occ_head(multi_scale, target_size=(self.nx, self.ny))

        return {
            "predictions": predictions,  # list of (B, 1, nx, ny) logits
            "hidden": hidden.detach(),   # for next timestep
            "depth": depth_dist,         # for depth supervision if needed
        }

    @torch.no_grad()
    def predict(self, batch, prev_hidden=None, threshold=0.5):
        """
        Inference: returns binary occupancy grid.
        
        Returns:
            occ_grid: (B, nx, ny) binary occupancy
            occ_prob: (B, nx, ny) occupancy probability
        """
        self.eval()
        output = self.forward(batch, prev_hidden)

        # Use finest scale prediction
        logits = output["predictions"][0]  # (B, 1, nx, ny)
        prob = torch.sigmoid(logits).squeeze(1)  # (B, nx, ny)
        binary = (prob > threshold).float()

        return binary, prob, output["hidden"]
