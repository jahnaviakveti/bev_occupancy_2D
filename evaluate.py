#!/usr/bin/env python3
# evaluate.py
"""
Evaluation script for BEV-Occ model.

Computes:
- Occupancy IoU (global + per-distance-bin)
- Distance-weighted Error
- Precision / Recall
- Inference speed (FPS)

Usage:
    python evaluate.py --config configs/bevocc_efficientb4.py --checkpoint checkpoints/best.pth
"""

import argparse
import importlib.util
import time

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from models.bevocc import BEVOcc
from utils.metrics import OccupancyMetrics


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module.config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Build model
    model = BEVOcc(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Build dataset
    from nuscenes.nuscenes import NuScenes
    from data.nuscenes_dataset import NuScenesBEVDataset, collate_fn

    data_cfg = cfg["data"]
    nusc = NuScenes(version=data_cfg["version"], dataroot=data_cfg["dataroot"], verbose=True)
    val_dataset = NuScenesBEVDataset(nusc, split="val", config=data_cfg)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # Metrics
    metrics = OccupancyMetrics(
        x_bound=tuple(cfg["model"]["bev_x_bound"]),
        y_bound=tuple(cfg["model"]["bev_y_bound"]),
        threshold=args.threshold,
    )

    # Evaluate
    total_time = 0.0
    num_frames = 0

    print("Running evaluation...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            intrinsics = batch["intrinsics"].to(device)
            extrinsics = batch["extrinsics"].to(device)
            bev_gt = batch["bev_gt"].to(device)

            model_input = {
                "images": images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
            }

            # Time inference
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()

            with autocast(enabled=True):
                output = model(model_input)

            torch.cuda.synchronize() if device.type == "cuda" else None
            total_time += time.time() - t0
            num_frames += images.shape[0]

            # Compute metrics
            pred_prob = torch.sigmoid(output["predictions"][0].squeeze(1))
            metrics.update(pred_prob, bev_gt)

    # Print results
    fps = num_frames / max(total_time, 1e-6)
    print(f"\nInference Speed: {fps:.1f} FPS ({total_time / num_frames * 1000:.1f} ms/frame)")
    print(metrics.summary())

    # Save results
    results = metrics.compute()
    results["fps"] = fps
    print(f"\nResults saved. Key metrics:")
    print(f"  Occupancy IoU:           {results['occupancy_iou']:.4f}")
    print(f"  Distance-Weighted Error: {results['distance_weighted_error']:.4f}")
    print(f"  FPS:                     {fps:.1f}")


if __name__ == "__main__":
    main()
