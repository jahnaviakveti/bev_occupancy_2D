#!/usr/bin/env python3
# inference.py
"""
Inference and visualization for BEV-Occ model.

Generates BEV occupancy predictions and creates visualizations
showing camera images alongside the predicted occupancy grid.

Usage:
    python inference.py --config configs/bevocc_efficientb4.py \
                        --checkpoint checkpoints/best.pth \
                        --sample-token <token>
    
    # Visualize random samples
    python inference.py --config configs/bevocc_efficientb4.py \
                        --checkpoint checkpoints/best.pth \
                        --num-samples 10 --output-dir vis/
"""

import argparse
import importlib.util
import os
import numpy as np
import torch
from torch.cuda.amp import autocast

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from models.bevocc import BEVOcc


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module.config


def visualize_bev(pred_prob, gt=None, save_path=None, title="BEV Occupancy"):
    """
    Visualize BEV occupancy prediction as a heatmap.
    
    Args:
        pred_prob: (H, W) numpy array, occupancy probability [0, 1]
        gt: (H, W) numpy array, ground truth binary (optional)
        save_path: path to save figure
        title: plot title
    """
    fig, axes = plt.subplots(1, 2 if gt is not None else 1, figsize=(12, 6))
    if gt is None:
        axes = [axes]

    # Custom colormap: blue (free) → yellow → red (occupied)
    colors = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ff6b6b"]
    cmap = LinearSegmentedColormap.from_list("occ", colors)

    # Prediction
    ax = axes[0]
    im = ax.imshow(pred_prob, cmap=cmap, vmin=0, vmax=1, origin="lower")
    ax.set_title(f"{title} - Prediction")
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")

    # Add ego vehicle marker at center
    cx, cy = pred_prob.shape[1] // 2, pred_prob.shape[0] // 2
    ax.plot(cx, cy, "w^", markersize=10, label="Ego")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Ground truth
    if gt is not None:
        ax = axes[1]
        im = ax.imshow(gt, cmap=cmap, vmin=0, vmax=1, origin="lower")
        ax.set_title(f"{title} - Ground Truth")
        ax.set_xlabel("X (grid cells)")
        ax.set_ylabel("Y (grid cells)")
        ax.plot(cx, cy, "w^", markersize=10, label="Ego")
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def visualize_full(images, pred_prob, gt=None, save_path=None, camera_names=None):
    """
    Full visualization: camera images + BEV prediction + GT.
    
    Args:
        images: list of (H, W, 3) numpy arrays (camera images, unnormalized)
        pred_prob: (H, W) BEV prediction
        gt: (H, W) BEV ground truth
        save_path: path to save
    """
    n_cams = len(images)

    fig = plt.figure(figsize=(20, 12))

    # Top row: camera images (3 front cameras)
    for i in range(min(3, n_cams)):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(images[i])
        name = camera_names[i] if camera_names else f"Camera {i}"
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    # Middle row: camera images (3 back cameras)
    for i in range(3, min(6, n_cams)):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(images[i])
        name = camera_names[i] if camera_names else f"Camera {i}"
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    # Bottom row: BEV prediction and GT
    colors = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ff6b6b"]
    cmap = LinearSegmentedColormap.from_list("occ", colors)

    ax = fig.add_subplot(3, 3, 7)
    ax.imshow(pred_prob, cmap=cmap, vmin=0, vmax=1, origin="lower")
    ax.set_title("BEV Prediction", fontsize=10)
    cx, cy = pred_prob.shape[1] // 2, pred_prob.shape[0] // 2
    ax.plot(cx, cy, "w^", markersize=8)

    if gt is not None:
        ax = fig.add_subplot(3, 3, 8)
        ax.imshow(gt, cmap=cmap, vmin=0, vmax=1, origin="lower")
        ax.set_title("Ground Truth", fontsize=10)
        ax.plot(cx, cy, "w^", markersize=8)

        # Difference map
        ax = fig.add_subplot(3, 3, 9)
        diff = np.abs(pred_prob - gt)
        ax.imshow(diff, cmap="hot", vmin=0, vmax=1, origin="lower")
        ax.set_title("Error Map", fontsize=10)
        ax.plot(cx, cy, "w^", markersize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-token", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="vis")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    model = BEVOcc(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build dataset
    from nuscenes.nuscenes import NuScenes
    from data.nuscenes_dataset import NuScenesBEVDataset, collate_fn

    data_cfg = cfg["data"]
    nusc = NuScenes(version=data_cfg["version"], dataroot=data_cfg["dataroot"], verbose=True)
    val_dataset = NuScenesBEVDataset(nusc, split="val", config=data_cfg)

    # Select samples
    if args.sample_token:
        indices = [i for i, t in enumerate(val_dataset.samples)
                   if t == args.sample_token]
    else:
        import random
        indices = random.sample(range(len(val_dataset)), min(args.num_samples, len(val_dataset)))

    camera_names = NuScenesBEVDataset.CAMERAS

    print(f"Visualizing {len(indices)} samples...")
    for idx in indices:
        sample = val_dataset[idx]
        token = sample["sample_token"]

        # Prepare batch
        batch = {
            "images": sample["images"].unsqueeze(0).to(device),
            "intrinsics": sample["intrinsics"].unsqueeze(0).to(device),
            "extrinsics": sample["extrinsics"].unsqueeze(0).to(device),
        }

        # Predict
        with torch.no_grad(), autocast(enabled=True):
            output = model(batch)

        pred_prob = torch.sigmoid(output["predictions"][0]).squeeze().cpu().numpy()
        gt = sample["bev_gt"].numpy()

        # Unnormalize images for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        imgs = []
        for i in range(sample["images"].shape[0]):
            img = sample["images"][i].permute(1, 2, 0).numpy()
            img = img * std + mean
            img = np.clip(img, 0, 1)
            imgs.append(img)

        # Visualize
        save_path = os.path.join(args.output_dir, f"{token[:8]}_full.png")
        visualize_full(imgs, pred_prob, gt, save_path, camera_names)

        save_path = os.path.join(args.output_dir, f"{token[:8]}_bev.png")
        visualize_bev(pred_prob, gt, save_path)

    print(f"Done! Visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
