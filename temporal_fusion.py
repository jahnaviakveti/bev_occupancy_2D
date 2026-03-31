# models/temporal_fusion.py
"""
Temporal BEV Fusion Module.

Warps the previous timestep's BEV features using ego-motion,
then fuses with current BEV features using a ConvGRU.
This improves occupancy prediction in occluded regions
and provides temporal consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """2D Convolutional GRU cell for BEV feature fusion."""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.reset_gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid(),
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x, h_prev):
        """
        Args:
            x: (B, C_in, H, W) current BEV features
            h_prev: (B, C_h, H, W) previous hidden state (warped)
        Returns:
            h_new: (B, C_h, H, W) updated hidden state
        """
        combined = torch.cat([x, h_prev], dim=1)
        r = self.reset_gate(combined)
        z = self.update_gate(combined)

        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_candidate = self.candidate(combined_r)

        h_new = (1 - z) * h_prev + z * h_candidate
        return h_new


class TemporalFusion(nn.Module):
    """
    Temporal BEV feature fusion with ego-motion compensation.
    
    Pipeline:
    1. Take previous BEV features
    2. Warp them to current frame using ego-motion transform
    3. Fuse warped features with current features via ConvGRU
    """

    def __init__(self, bev_dim=64, hidden_dim=64,
                 x_bound=(-50.0, 50.0, 0.5),
                 y_bound=(-50.0, 50.0, 0.5)):
        super().__init__()
        self.bev_dim = bev_dim
        self.hidden_dim = hidden_dim
        self.x_bound = x_bound
        self.y_bound = y_bound

        self.nx = int((x_bound[1] - x_bound[0]) / x_bound[2])
        self.ny = int((y_bound[1] - y_bound[0]) / y_bound[2])

        # ConvGRU for temporal fusion
        self.gru = ConvGRUCell(bev_dim, hidden_dim)

        # Project input BEV features to match hidden dim if needed
        if bev_dim != hidden_dim:
            self.input_proj = nn.Conv2d(bev_dim, hidden_dim, 1)
        else:
            self.input_proj = nn.Identity()

    def warp_bev(self, bev_feat, ego_motion):
        """
        Warp BEV features from previous frame to current frame
        using ego-motion transformation.
        
        Args:
            bev_feat: (B, C, H, W) BEV features from previous frame
            ego_motion: (B, 4, 4) transformation matrix (prev_ego → curr_ego)
        
        Returns:
            warped: (B, C, H, W) warped BEV features
        """
        B, C, H, W = bev_feat.shape
        device = bev_feat.device

        # Create grid of BEV coordinates
        # Each pixel (i, j) corresponds to a real-world (x, y) position
        xs = torch.linspace(
            self.x_bound[0] + self.x_bound[2] / 2,
            self.x_bound[1] - self.x_bound[2] / 2,
            self.nx, device=device
        )
        ys = torch.linspace(
            self.y_bound[0] + self.y_bound[2] / 2,
            self.y_bound[1] - self.y_bound[2] / 2,
            self.ny, device=device
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Create homogeneous coordinates (x, y, 0, 1) for ground plane
        ones = torch.ones_like(grid_x)
        zeros = torch.zeros_like(grid_x)
        coords = torch.stack([grid_x, grid_y, zeros, ones], dim=-1)  # (H, W, 4)
        coords = coords.view(-1, 4)  # (H*W, 4)

        # Apply inverse ego-motion to find where each current pixel
        # was in the previous frame
        # curr_point = T @ prev_point  →  prev_point = T_inv @ curr_point
        ego_inv = torch.inverse(ego_motion)  # (B, 4, 4)
        prev_coords = torch.einsum("bij,pj->bpi", ego_inv, coords)  # (B, H*W, 4)

        # Convert back to grid indices (normalized to [-1, 1] for grid_sample)
        prev_x = prev_coords[..., 0]  # (B, H*W)
        prev_y = prev_coords[..., 1]

        # Normalize to [-1, 1]
        norm_x = (prev_x - self.x_bound[0]) / (self.x_bound[1] - self.x_bound[0]) * 2 - 1
        norm_y = (prev_y - self.y_bound[0]) / (self.y_bound[1] - self.y_bound[0]) * 2 - 1

        # Create sampling grid
        grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, H*W, 2)
        grid = grid.view(B, H, W, 2)

        # Warp using bilinear interpolation
        warped = F.grid_sample(
            bev_feat, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True
        )

        return warped

    def forward(self, current_bev, prev_hidden, ego_motion=None):
        """
        Fuse current BEV features with temporal context.
        
        Args:
            current_bev: (B, C, H, W) current frame BEV features
            prev_hidden: (B, C_h, H, W) previous hidden state, or None
            ego_motion: (B, 4, 4) prev_ego → curr_ego transform, or None
        
        Returns:
            fused: (B, C_h, H, W) temporally fused BEV features
        """
        B = current_bev.shape[0]
        device = current_bev.device

        current_proj = self.input_proj(current_bev)

        if prev_hidden is None or ego_motion is None:
            # First frame: initialize hidden state from current features
            return current_proj

        # Warp previous hidden state to current frame
        warped_hidden = self.warp_bev(prev_hidden, ego_motion)

        # Fuse via ConvGRU
        fused = self.gru(current_proj, warped_hidden)

        return fused
