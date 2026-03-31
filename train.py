#!/usr/bin/env python3
# train.py
"""
Training script for BEV-Occ model.

Features:
- Mixed precision (AMP) training
- Gradient accumulation
- Cosine LR schedule with warmup
- Multi-scale deep supervision loss
- Distance-weighted BCE loss
- TensorBoard logging
- Checkpoint saving with best model tracking

Usage:
    python train.py --config configs/bevocc_efficientb4.py
    python train.py --config configs/bevocc_efficientb4.py --resume checkpoints/latest.pth
"""

import argparse
import importlib.util
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from models.bevocc import BEVOcc
from utils.losses import MultiScaleLoss
from utils.metrics import OccupancyMetrics


def load_config(config_path):
    """Load config from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module.config


def build_optimizer(model, cfg):
    """Build optimizer with separate LR for backbone."""
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": cfg["lr"] * 0.1},  # Lower LR for pretrained
        {"params": other_params, "lr": cfg["lr"]},
    ]

    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    else:
        return torch.optim.Adam(param_groups, weight_decay=cfg["weight_decay"])


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Build cosine LR scheduler with warmup."""
    total_steps = cfg["epochs"] * steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        else:
            import math
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(cfg["min_lr"] / cfg["lr"],
                       0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    scaler, device, epoch, cfg, writer=None):
    """Train for one epoch."""
    model.train()
    metrics = OccupancyMetrics()

    accum_steps = cfg["accumulation_steps"]
    total_loss = 0.0
    num_batches = 0
    prev_hidden = None

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        bev_gt = batch["bev_gt"].to(device)
        ego_motion = batch["ego_motion"].to(device)

        model_input = {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "ego_motion": ego_motion,
        }

        # Forward pass with mixed precision
        with autocast(enabled=cfg.get("use_amp", True)):
            output = model(model_input, prev_hidden=None)  # No temporal in training for simplicity
            
            # Target shape: (B, 1, H, W)
            target = bev_gt.unsqueeze(1)
            loss, loss_dict = criterion(output["predictions"], target)
            loss = loss / accum_steps

        # Backward
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accum_steps == 0:
            if cfg.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Track metrics
        total_loss += loss_dict["total_loss"]
        num_batches += 1

        with torch.no_grad():
            pred_prob = torch.sigmoid(output["predictions"][0].squeeze(1))
            metrics.update(pred_prob, bev_gt)

        # Logging
        if batch_idx % 50 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            lr = optimizer.param_groups[1]["lr"]
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_dict['total_loss']:.4f} LR: {lr:.6f}")

            if writer:
                writer.add_scalar("train/loss", loss_dict["total_loss"], global_step)
                writer.add_scalar("train/lr", lr, global_step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v, global_step)

    avg_loss = total_loss / max(num_batches, 1)
    results = metrics.compute()

    return avg_loss, results


@torch.no_grad()
def validate(model, dataloader, criterion, device, cfg):
    """Validate the model."""
    model.eval()
    metrics = OccupancyMetrics()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        bev_gt = batch["bev_gt"].to(device)

        model_input = {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

        with autocast(enabled=cfg.get("use_amp", True)):
            output = model(model_input)
            target = bev_gt.unsqueeze(1)
            loss, loss_dict = criterion(output["predictions"], target)

        total_loss += loss_dict["total_loss"]
        num_batches += 1

        pred_prob = torch.sigmoid(output["predictions"][0].squeeze(1))
        metrics.update(pred_prob, bev_gt)

    avg_loss = total_loss / max(num_batches, 1)
    results = metrics.compute()

    return avg_loss, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = BEVOcc(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    # Build dataset & dataloader
    from nuscenes.nuscenes import NuScenes
    from data.nuscenes_dataset import NuScenesBEVDataset, collate_fn

    nusc = NuScenes(version=data_cfg["version"], dataroot=data_cfg["dataroot"], verbose=True)

    train_dataset = NuScenesBEVDataset(nusc, split=data_cfg["train_split"], config=data_cfg)
    val_dataset = NuScenesBEVDataset(nusc, split=data_cfg["val_split"], config=data_cfg)

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn, pin_memory=True,
    )

    # Loss
    criterion = MultiScaleLoss(
        x_bound=tuple(model_cfg["bev_x_bound"]),
        y_bound=tuple(model_cfg["bev_y_bound"]),
        alpha=train_cfg["distance_weight_alpha"],
        pos_weight=train_cfg["bce_pos_weight"],
        scale_weights=tuple(train_cfg["deep_supervision_weights"]),
    )

    # Optimizer & scheduler
    optimizer = build_optimizer(model, train_cfg)
    steps_per_epoch = len(train_loader) // train_cfg["accumulation_steps"]
    scheduler = build_scheduler(optimizer, train_cfg, steps_per_epoch)

    # Mixed precision scaler
    scaler = GradScaler(enabled=train_cfg.get("use_amp", True))

    # TensorBoard
    writer = SummaryWriter("runs/bevocc")

    # Resume
    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", 0.0)
        print(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")

    # Checkpoint dir
    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)

    # Training loop
    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, train_cfg, writer
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, train_cfg
        )

        elapsed = time.time() - t0

        # Log
        print(f"\nEpoch {epoch} ({elapsed:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f}  IoU: {train_metrics['occupancy_iou']:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  IoU: {val_metrics['occupancy_iou']:.4f}  "
              f"DWE: {val_metrics['distance_weighted_error']:.4f}")

        if writer:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/iou", val_metrics["occupancy_iou"], epoch)
            writer.add_scalar("val/dwe", val_metrics["distance_weighted_error"], epoch)

        # Save checkpoint
        is_best = val_metrics["occupancy_iou"] > best_iou
        if is_best:
            best_iou = val_metrics["occupancy_iou"]

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": best_iou,
            "val_metrics": val_metrics,
        }

        if (epoch + 1) % train_cfg["save_every"] == 0:
            torch.save(ckpt, os.path.join(train_cfg["checkpoint_dir"], f"epoch_{epoch}.pth"))
        torch.save(ckpt, os.path.join(train_cfg["checkpoint_dir"], "latest.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(train_cfg["checkpoint_dir"], "best.pth"))
            print(f"  ★ New best IoU: {best_iou:.4f}")

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
