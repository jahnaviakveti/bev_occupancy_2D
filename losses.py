# utils/losses.py
"""
Custom loss functions for BEV occupancy prediction.

Key innovation: Distance-weighted BCE loss that penalizes errors
closer to the ego-vehicle more heavily, directly aligned with
the competition evaluation metric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceWeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss weighted by distance from ego-vehicle.
    
    Errors at close range (safety-critical) are penalized more than
    errors far away. Weight = 1 / (distance ^ alpha).
    
    This directly optimizes the "Distance-weighted Error" metric
    specified in the problem statement.
    """

    def __init__(self, x_bound=(-50.0, 50.0, 0.5),
                 y_bound=(-50.0, 50.0, 0.5),
                 alpha=1.0, pos_weight=2.5,
                 min_distance=1.0):
        super().__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.min_distance = min_distance

        # Precompute distance weight map
        nx = int((x_bound[1] - x_bound[0]) / x_bound[2])
        ny = int((y_bound[1] - y_bound[0]) / y_bound[2])

        # Create coordinate grid
        xs = torch.linspace(
            x_bound[0] + x_bound[2] / 2,
            x_bound[1] - x_bound[2] / 2,
            nx
        )
        ys = torch.linspace(
            y_bound[0] + y_bound[2] / 2,
            y_bound[1] - y_bound[2] / 2,
            ny
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # Distance from ego (origin)
        distance = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        distance = distance.clamp(min=min_distance)

        # Weight: inverse distance
        weight_map = 1.0 / (distance ** alpha)

        # Normalize so mean weight = 1
        weight_map = weight_map / weight_map.mean()

        # Register as buffer (moves with model to GPU, not a parameter)
        self.register_buffer("weight_map", weight_map.unsqueeze(0).unsqueeze(0))

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits
            target: (B, 1, H, W) binary ground truth
        Returns:
            loss: scalar
        """
        # Apply pos_weight for class imbalance
        pos_weight = torch.tensor([self.pos_weight], device=pred_logits.device)

        # Per-pixel BCE
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target,
            pos_weight=pos_weight,
            reduction="none"
        )  # (B, 1, H, W)

        # Apply distance weighting
        weight = self.weight_map.to(pred_logits.device)

        # Handle case where pred and weight sizes differ (multi-scale)
        if bce.shape[-2:] != weight.shape[-2:]:
            weight = F.interpolate(
                weight, size=bce.shape[-2:],
                mode="bilinear", align_corners=True
            )

        weighted_bce = bce * weight

        return weighted_bce.mean()


class MultiScaleLoss(nn.Module):
    """
    Deep supervision loss: compute loss at each scale and weight them.
    
    Coarser scales stabilize training (easier gradients),
    fine scale provides precise localization.
    """

    def __init__(self, x_bound=(-50.0, 50.0, 0.5),
                 y_bound=(-50.0, 50.0, 0.5),
                 alpha=1.0, pos_weight=2.5,
                 scale_weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.scale_weights = scale_weights

        # One loss per scale (they all use the same distance weighting)
        self.loss_fn = DistanceWeightedBCELoss(
            x_bound=x_bound, y_bound=y_bound,
            alpha=alpha, pos_weight=pos_weight,
        )

    def forward(self, predictions, target):
        """
        Args:
            predictions: list of (B, 1, H, W) logits at different scales
                        (all already upsampled to target size by OccupancyHead)
            target: (B, 1, H, W) binary ground truth
        Returns:
            total_loss: scalar
            loss_dict: dict of individual scale losses
        """
        total_loss = 0.0
        loss_dict = {}

        for i, (pred, w) in enumerate(zip(predictions, self.scale_weights)):
            scale_loss = self.loss_fn(pred, target)
            total_loss = total_loss + w * scale_loss
            loss_dict[f"loss_scale_{i}"] = scale_loss.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


class DepthSupervisionLoss(nn.Module):
    """
    Optional: supervise depth prediction using LiDAR projected depths.
    Improves the quality of the Lift step.
    """

    def __init__(self, depth_min=1.0, depth_max=57.0, depth_channels=112):
        super().__init__()
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_channels = depth_channels

    def forward(self, pred_depth, gt_depth_map, valid_mask):
        """
        Args:
            pred_depth: (B*N, D, H, W) predicted depth distribution
            gt_depth_map: (B*N, H, W) ground truth depth from projected LiDAR
            valid_mask: (B*N, H, W) binary mask of pixels with valid depth
        Returns:
            loss: scalar
        """
        # Convert GT depth to bin index
        bin_size = (self.depth_max - self.depth_min) / self.depth_channels
        gt_bins = ((gt_depth_map - self.depth_min) / bin_size).long()
        gt_bins = gt_bins.clamp(0, self.depth_channels - 1)

        # Create one-hot target
        gt_onehot = torch.zeros_like(pred_depth)
        gt_onehot.scatter_(1, gt_bins.unsqueeze(1), 1.0)

        # Cross-entropy loss on valid pixels only
        log_pred = torch.log(pred_depth + 1e-8)
        loss = -(gt_onehot * log_pred).sum(dim=1)  # (B*N, H, W)

        if valid_mask.sum() > 0:
            loss = (loss * valid_mask).sum() / valid_mask.sum()
        else:
            loss = loss.mean() * 0.0

        return loss
