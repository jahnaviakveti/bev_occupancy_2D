# utils/metrics.py
"""
Evaluation metrics for BEV occupancy prediction.

Metrics:
1. Occupancy IoU — standard Intersection-over-Union
2. Distance-weighted Error — errors weighted by proximity to ego
3. Per-distance-bin IoU — breakdown by range
"""

import torch
import numpy as np


class OccupancyMetrics:
    """Compute and accumulate occupancy evaluation metrics."""

    def __init__(self, x_bound=(-50.0, 50.0, 0.5),
                 y_bound=(-50.0, 50.0, 0.5),
                 distance_bins=(0, 10, 20, 30, 40, 50),
                 threshold=0.5):
        self.threshold = threshold
        self.distance_bins = distance_bins
        self.x_bound = x_bound
        self.y_bound = y_bound

        # Precompute distance map
        nx = int((x_bound[1] - x_bound[0]) / x_bound[2])
        ny = int((y_bound[1] - y_bound[0]) / y_bound[2])

        xs = np.linspace(
            x_bound[0] + x_bound[2] / 2,
            x_bound[1] - x_bound[2] / 2,
            nx
        )
        ys = np.linspace(
            y_bound[0] + y_bound[2] / 2,
            y_bound[1] - y_bound[2] / 2,
            ny
        )
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        self.distance_map = np.sqrt(grid_x ** 2 + grid_y ** 2)

        self.reset()

    def reset(self):
        """Reset accumulators."""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.total_weighted_error = 0.0
        self.total_weight = 0.0
        self.n_samples = 0

        # Per-distance-bin accumulators
        self.bin_tp = {b: 0 for b in range(len(self.distance_bins) - 1)}
        self.bin_fp = {b: 0 for b in range(len(self.distance_bins) - 1)}
        self.bin_fn = {b: 0 for b in range(len(self.distance_bins) - 1)}

    def update(self, pred_prob, target):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred_prob: (B, H, W) predicted occupancy probability [0, 1]
            target: (B, H, W) binary ground truth {0, 1}
        """
        if isinstance(pred_prob, torch.Tensor):
            pred_prob = pred_prob.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        pred_binary = (pred_prob > self.threshold).astype(np.float32)

        B = pred_prob.shape[0]
        for b in range(B):
            p = pred_binary[b]
            t = target[b]

            # Global IoU components
            self.tp += np.sum((p == 1) & (t == 1))
            self.fp += np.sum((p == 1) & (t == 0))
            self.fn += np.sum((p == 0) & (t == 1))

            # Distance-weighted error
            error = np.abs(p - t)
            dist = np.maximum(self.distance_map, 1.0)
            weight = 1.0 / dist
            self.total_weighted_error += np.sum(error * weight)
            self.total_weight += np.sum(weight)

            # Per-distance-bin IoU
            for i in range(len(self.distance_bins) - 1):
                d_min = self.distance_bins[i]
                d_max = self.distance_bins[i + 1]
                mask = (self.distance_map >= d_min) & (self.distance_map < d_max)

                p_bin = p[mask]
                t_bin = t[mask]
                self.bin_tp[i] += np.sum((p_bin == 1) & (t_bin == 1))
                self.bin_fp[i] += np.sum((p_bin == 1) & (t_bin == 0))
                self.bin_fn[i] += np.sum((p_bin == 0) & (t_bin == 1))

        self.n_samples += B

    def compute(self):
        """
        Compute final metrics.
        
        Returns:
            results: dict with all metrics
        """
        results = {}

        # Global IoU
        iou = self.tp / max(self.tp + self.fp + self.fn, 1)
        results["occupancy_iou"] = iou

        # Distance-weighted error
        dwe = self.total_weighted_error / max(self.total_weight, 1)
        results["distance_weighted_error"] = dwe

        # Precision & Recall
        results["precision"] = self.tp / max(self.tp + self.fp, 1)
        results["recall"] = self.tp / max(self.tp + self.fn, 1)

        # Per-distance-bin IoU
        for i in range(len(self.distance_bins) - 1):
            d_min = self.distance_bins[i]
            d_max = self.distance_bins[i + 1]
            bin_iou = self.bin_tp[i] / max(
                self.bin_tp[i] + self.bin_fp[i] + self.bin_fn[i], 1
            )
            results[f"iou_{d_min}-{d_max}m"] = bin_iou

        results["n_samples"] = self.n_samples

        return results

    def summary(self):
        """Pretty-print metrics."""
        results = self.compute()
        lines = [
            "=" * 50,
            "BEV Occupancy Evaluation Results",
            "=" * 50,
            f"  Occupancy IoU:           {results['occupancy_iou']:.4f}",
            f"  Distance-Weighted Error: {results['distance_weighted_error']:.4f}",
            f"  Precision:               {results['precision']:.4f}",
            f"  Recall:                  {results['recall']:.4f}",
            "-" * 50,
            "  Per-Distance IoU:",
        ]
        for i in range(len(self.distance_bins) - 1):
            d_min = self.distance_bins[i]
            d_max = self.distance_bins[i + 1]
            key = f"iou_{d_min}-{d_max}m"
            lines.append(f"    {d_min:2d}-{d_max:2d}m: {results[key]:.4f}")
        lines.append(f"  Total samples: {results['n_samples']}")
        lines.append("=" * 50)
        return "\n".join(lines)
