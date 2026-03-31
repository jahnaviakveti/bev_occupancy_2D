# configs/bevocc_efficientb4.py
"""
Configuration for BEV-Occ model.
All hyperparameters and dataset settings in one place.
"""

config = dict(
    # ── Model ──────────────────────────────────────────────
    model=dict(
        backbone="efficientnet-b4",
        # Image feature dimensions (EfficientNet-B4 outputs 160 channels at stride 8)
        image_feat_dim=160,
        # Depth prediction
        depth_channels=112,          # Number of depth bins
        depth_min=1.0,               # meters
        depth_max=57.0,              # meters
        # BEV grid
        bev_x_bound=[-50.0, 50.0, 0.5],   # [min, max, resolution] in meters
        bev_y_bound=[-50.0, 50.0, 0.5],   # 200x200 grid
        bev_z_bound=[-5.0, 3.0, 8.0],     # [min, max, pillar_height]
        bev_feat_dim=64,
        # BEV encoder
        bev_encoder_channels=[64, 128, 256],
        # Temporal fusion
        use_temporal=True,
        temporal_hidden_dim=64,
        # Occupancy head
        occupancy_classes=1,         # binary: occupied vs free
    ),

    # ── Dataset ────────────────────────────────────────────
    data=dict(
        dataroot="data/nuscenes",
        version="v1.0-trainval",     # or "v1.0-mini" for debugging
        train_split="train",
        val_split="val",
        # Camera selection
        cameras=[
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        ],
        # Image preprocessing
        input_size=(256, 704),       # H, W after resize
        # Augmentation
        use_color_jitter=True,
        use_grid_mask=True,          # GridMask augmentation on BEV
        # Ground truth
        gt_dir="data/nuscenes/bev_gt",
        num_sweeps=10,               # LiDAR sweeps for GT generation
    ),

    # ── Training ───────────────────────────────────────────
    training=dict(
        epochs=24,
        batch_size=4,                # per GPU
        num_workers=4,
        # Optimizer
        optimizer="adamw",
        lr=2e-4,
        weight_decay=1e-2,
        # Scheduler
        scheduler="cosine",
        warmup_epochs=2,
        min_lr=1e-6,
        # Mixed precision
        use_amp=True,
        # Gradient
        grad_clip=5.0,
        accumulation_steps=2,        # effective batch = 4 * 2 = 8
        # Loss
        distance_weight_alpha=1.0,   # 1/d^alpha weighting
        bce_pos_weight=2.5,          # handle class imbalance (more free than occupied)
        # Deep supervision weights for multi-scale
        deep_supervision_weights=[1.0, 0.5, 0.25],
        # Checkpointing
        save_every=2,
        checkpoint_dir="checkpoints",
    ),

    # ── Evaluation ─────────────────────────────────────────
    evaluation=dict(
        iou_threshold=0.5,
        distance_bins=[0, 10, 20, 30, 40, 50],  # meters
        use_tta=False,               # test-time augmentation
    ),
)
