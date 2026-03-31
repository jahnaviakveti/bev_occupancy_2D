# models/bev_transform.py
"""
BEV Transform: Lift-Splat-Shoot view transformation.

Lifts 2D image features into 3D using predicted depth distributions,
then splats them onto the BEV plane via voxel pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiftSplat(nn.Module):
    """
    Lift-Splat view transformation.
    
    1. Create a frustum of 3D points for each camera
    2. Weight image features by depth distribution (Lift)
    3. Project weighted features into BEV voxel grid (Splat)
    4. Collapse height dimension to get BEV features
    """

    def __init__(self, feat_dim=160, depth_channels=112,
                 x_bound=(-50.0, 50.0, 0.5),
                 y_bound=(-50.0, 50.0, 0.5),
                 z_bound=(-5.0, 3.0, 8.0),
                 depth_min=1.0, depth_max=57.0):
        super().__init__()

        self.feat_dim = feat_dim
        self.depth_channels = depth_channels
        self.depth_min = depth_min
        self.depth_max = depth_max

        # BEV grid parameters
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound

        # Grid dimensions
        self.nx = int((x_bound[1] - x_bound[0]) / x_bound[2])  # 200
        self.ny = int((y_bound[1] - y_bound[0]) / y_bound[2])  # 200
        self.nz = int((z_bound[1] - z_bound[0]) / z_bound[2])  # 1

        # Depth bins (uniform spacing)
        self.depth_bins = torch.linspace(
            depth_min, depth_max, depth_channels
        )

        # Frustum will be created on first forward pass
        self.frustum = None

    def create_frustum(self, feat_h, feat_w, device):
        """
        Create a frustum of 3D points in camera coordinate frame.
        
        Returns:
            frustum: (D, H, W, 3) - 3D coordinates for each depth/pixel combo
        """
        depth_bins = self.depth_bins.to(device)

        # Create meshgrid of pixel coordinates
        # These are in the feature map resolution (not original image)
        ys = torch.arange(feat_h, device=device).float()
        xs = torch.arange(feat_w, device=device).float()
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")

        # Expand to depth dimension: (D, H, W)
        ds = depth_bins.view(-1, 1, 1).expand(-1, feat_h, feat_w)
        xs = xs.unsqueeze(0).expand(self.depth_channels, -1, -1)
        ys = ys.unsqueeze(0).expand(self.depth_channels, -1, -1)

        # Stack to get frustum: (D, H, W, 3) with (x_pixel, y_pixel, depth)
        frustum = torch.stack([xs, ys, ds], dim=-1)

        return frustum

    def get_geometry(self, intrinsics, extrinsics, feat_h, feat_w, img_h, img_w):
        """
        Convert frustum pixel coordinates to ego-vehicle frame coordinates.
        
        Args:
            intrinsics: (B, N, 3, 3) camera intrinsic matrices
            extrinsics: (B, N, 4, 4) camera-to-ego transformation matrices
            feat_h, feat_w: feature map spatial dimensions
            img_h, img_w: original image dimensions
        
        Returns:
            geom: (B, N, D, H, W, 3) 3D points in ego frame
        """
        device = intrinsics.device
        B, N = intrinsics.shape[:2]

        if self.frustum is None or self.frustum.device != device:
            self.frustum = self.create_frustum(feat_h, feat_w, device)

        frustum = self.frustum.clone()  # (D, H, W, 3)

        # Scale pixel coordinates from feature map to original image resolution
        scale_x = img_w / feat_w
        scale_y = img_h / feat_h
        frustum[..., 0] *= scale_x
        frustum[..., 1] *= scale_y

        # Unproject: pixel coords + depth → camera 3D coords
        # p_cam = K^{-1} @ [u*d, v*d, d]^T
        D, fH, fW, _ = frustum.shape
        points = frustum.view(-1, 3).clone()  # (D*H*W, 3)

        # Convert to homogeneous camera coordinates
        # [u, v, d] → [u*d, v*d, d]
        points[:, 0] *= points[:, 2]
        points[:, 1] *= points[:, 2]

        # Apply inverse intrinsics: (B, N, 3, 3) @ (D*H*W, 3, 1) → camera coords
        inv_intrinsics = torch.inverse(intrinsics)  # (B, N, 3, 3)
        points = points.unsqueeze(0).unsqueeze(0)  # (1, 1, D*H*W, 3)
        points = points.expand(B, N, -1, -1)       # (B, N, D*H*W, 3)
        
        # Matrix multiply: camera_coords = K_inv @ pixel_coords
        cam_coords = torch.einsum("bnij,bnpj->bnpi", inv_intrinsics, points)

        # Apply extrinsics: camera → ego frame
        # Add homogeneous coordinate
        ones = torch.ones_like(cam_coords[..., :1])
        cam_coords_h = torch.cat([cam_coords, ones], dim=-1)  # (B, N, D*H*W, 4)

        # Transform to ego frame
        ego_coords = torch.einsum(
            "bnij,bnpj->bnpi", extrinsics, cam_coords_h
        )  # (B, N, D*H*W, 4)

        # Reshape to (B, N, D, H, W, 3)
        geom = ego_coords[..., :3].view(B, N, D, fH, fW, 3)

        return geom

    def voxel_pooling(self, geom, features):
        """
        Splat features into BEV grid using the 3D geometry.
        
        Args:
            geom: (B, N, D, H, W, 3) 3D points in ego frame
            features: (B, N, D, H, W, C) depth-weighted features
        
        Returns:
            bev: (B, C, nx, ny) BEV feature map
        """
        B, N, D, H, W, C = features.shape
        device = features.device

        # Compute voxel indices for each point
        # X axis
        x_idx = ((geom[..., 0] - self.x_bound[0]) / self.x_bound[2]).long()
        # Y axis
        y_idx = ((geom[..., 1] - self.y_bound[0]) / self.y_bound[2]).long()

        # Valid mask: points that fall within the BEV grid
        valid = (
            (x_idx >= 0) & (x_idx < self.nx) &
            (y_idx >= 0) & (y_idx < self.ny)
        )

        # Flatten everything for scatter
        x_idx_flat = x_idx[valid]
        y_idx_flat = y_idx[valid]
        feats_flat = features[valid]  # (num_valid, C)

        # Compute linear index into BEV grid
        # We need batch index too
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1)
        batch_idx = batch_idx.expand(B, N, D, H, W)[valid]

        linear_idx = batch_idx * (self.nx * self.ny) + x_idx_flat * self.ny + y_idx_flat

        # Scatter-add features into BEV grid
        bev = torch.zeros(B * self.nx * self.ny, C, device=device)
        bev.scatter_add_(0, linear_idx.unsqueeze(-1).expand(-1, C), feats_flat)
        bev = bev.view(B, self.nx, self.ny, C)
        bev = bev.permute(0, 3, 1, 2)  # (B, C, nx, ny)

        return bev

    def forward(self, image_features, depth_dist, context_features,
                intrinsics, extrinsics, img_h, img_w):
        """
        Full Lift-Splat forward pass.
        
        Args:
            image_features: (B, N, C, H, W) backbone features
            depth_dist: (B, N, D, H, W) depth probability distribution
            context_features: (B, N, C, H, W) context features from DepthNet
            intrinsics: (B, N, 3, 3) camera intrinsics
            extrinsics: (B, N, 4, 4) camera-to-ego transforms
            img_h, img_w: original image dimensions
        
        Returns:
            bev_features: (B, C, nx, ny) BEV feature map
        """
        B, N, C, H, W = context_features.shape
        D = depth_dist.shape[2]

        # Step 1: Get 3D geometry
        geom = self.get_geometry(intrinsics, extrinsics, H, W, img_h, img_w)

        # Step 2: Lift — outer product of depth and context
        # depth: (B, N, D, H, W) → (B, N, D, H, W, 1)
        # context: (B, N, C, H, W) → (B, N, 1, H, W, C)
        context_perm = context_features.permute(0, 1, 3, 4, 2).unsqueeze(2)  # (B, N, 1, H, W, C)
        depth_exp = depth_dist.unsqueeze(-1)  # (B, N, D, H, W, 1)
        volume = depth_exp * context_perm  # (B, N, D, H, W, C)

        # Step 3: Splat — project into BEV
        bev_features = self.voxel_pooling(geom, volume)

        return bev_features
