#!/usr/bin/env python3
# scripts/sanity_check.py
"""
Sanity check: Verify the model builds, runs forward/backward,
and produces correct output shapes — using synthetic data (no nuScenes needed).

Run this FIRST to confirm everything works before training.

Usage:
    python scripts/sanity_check.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time


def test_backbone():
    """Test image backbone and depth net."""
    print("=" * 60)
    print("TEST 1: Backbone + DepthNet")
    print("=" * 60)

    from models.backbone import EfficientNetBackbone, DepthNet

    backbone = EfficientNetBackbone(out_channels=160, pretrained=False)
    depth_net = DepthNet(in_channels=160, depth_channels=112)

    # Simulate 1 batch × 2 cameras for memory
    x = torch.randn(2, 3, 128, 352)

    feats = backbone(x)
    print(f"  Input:    {x.shape}")
    print(f"  Features: {feats.shape}")
    assert feats.shape[1] == 160, f"Expected 160 channels, got {feats.shape[1]}"

    depth, context = depth_net(feats)
    print(f"  Depth:    {depth.shape} (sum per pixel ≈ {depth[0, :, 0, 0].sum().item():.3f})")
    print(f"  Context:  {context.shape}")
    assert depth.shape[1] == 112
    assert abs(depth[0, :, 0, 0].sum().item() - 1.0) < 0.01, "Depth should be softmax (sum=1)"

    params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"  Backbone params: {params:.1f}M")
    print("  ✓ PASSED\n")


def test_lift_splat():
    """Test BEV transform."""
    print("=" * 60)
    print("TEST 2: Lift-Splat BEV Transform")
    print("=" * 60)

    from models.bev_transform import LiftSplat

    B, N, C, D = 2, 6, 64, 112  # Use smaller C for speed
    fH, fW = 32, 88
    img_h, img_w = 256, 704

    lift_splat = LiftSplat(
        feat_dim=C, depth_channels=D,
        x_bound=(-50.0, 50.0, 0.5),
        y_bound=(-50.0, 50.0, 0.5),
        z_bound=(-5.0, 3.0, 8.0),
    )

    # Synthetic inputs
    feats = torch.randn(B, N, C, fH, fW)
    depth = torch.softmax(torch.randn(B, N, D, fH, fW), dim=2)
    context = torch.randn(B, N, C, fH, fW)

    # Realistic camera intrinsics
    K = torch.tensor([
        [1266.0, 0.0, 816.0],
        [0.0, 1266.0, 491.0],
        [0.0, 0.0, 1.0],
    ]).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    # Adjust for feature map scale
    K[:, :, 0, :] *= fW / img_w
    K[:, :, 1, :] *= fH / img_h

    # Simple extrinsics (cameras at different positions around ego)
    import math
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    for i in range(N):
        angle = (i / N) * 2 * math.pi
        extrinsics[:, i, 0, 3] = 1.5 * math.cos(angle)
        extrinsics[:, i, 1, 3] = 1.5 * math.sin(angle)
        extrinsics[:, i, 2, 3] = 1.5

    bev = lift_splat(feats, depth, context, K, extrinsics, img_h, img_w)
    print(f"  Input features: {feats.shape}")
    print(f"  Depth dist:     {depth.shape}")
    print(f"  BEV output:     {bev.shape}")
    assert bev.shape == (B, C, 200, 200), f"Expected (2, {C}, 200, 200), got {bev.shape}"
    print("  ✓ PASSED\n")


def test_temporal_fusion():
    """Test temporal fusion module."""
    print("=" * 60)
    print("TEST 3: Temporal Fusion")
    print("=" * 60)

    from models.temporal_fusion import TemporalFusion

    B, C = 2, 64
    H, W = 200, 200

    temporal = TemporalFusion(bev_dim=C, hidden_dim=C)

    current = torch.randn(B, C, H, W)
    prev_hidden = torch.randn(B, C, H, W)
    ego_motion = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
    # Add small translation
    ego_motion[:, 0, 3] = 0.5
    ego_motion[:, 1, 3] = 0.1

    # First frame (no previous)
    out1 = temporal(current, None, None)
    print(f"  First frame (no prev): {out1.shape}")

    # Subsequent frame
    out2 = temporal(current, prev_hidden, ego_motion)
    print(f"  With temporal:         {out2.shape}")

    assert out1.shape == (B, C, H, W)
    assert out2.shape == (B, C, H, W)
    assert not torch.allclose(out1, out2), "Temporal fusion should change output"
    print("  ✓ PASSED\n")


def test_bev_encoder():
    """Test BEV encoder and occupancy head."""
    print("=" * 60)
    print("TEST 4: BEV Encoder + Occupancy Head")
    print("=" * 60)

    from models.bev_encoder import BEVEncoder, OccupancyHead

    B, C = 2, 64
    H, W = 200, 200

    encoder = BEVEncoder(in_channels=C, channels=(64, 128, 256))
    occ_head = OccupancyHead(in_channels_list=(64, 128, 256), num_classes=1)

    x = torch.randn(B, C, H, W)

    multi_scale, fused = encoder(x)
    print(f"  Input:          {x.shape}")
    for i, f in enumerate(multi_scale):
        print(f"  Scale {i}:        {f.shape}")
    print(f"  Fused:          {fused.shape}")

    preds = occ_head(multi_scale, target_size=(H, W))
    for i, p in enumerate(preds):
        print(f"  Pred scale {i}:   {p.shape}")

    assert all(p.shape == (B, 1, H, W) for p in preds)
    print("  ✓ PASSED\n")


def test_losses():
    """Test loss functions."""
    print("=" * 60)
    print("TEST 5: Loss Functions")
    print("=" * 60)

    from utils.losses import MultiScaleLoss

    B = 2
    H, W = 200, 200

    criterion = MultiScaleLoss(
        x_bound=(-50.0, 50.0, 0.5),
        y_bound=(-50.0, 50.0, 0.5),
        alpha=1.0, pos_weight=2.5,
    )

    # Simulate multi-scale predictions
    preds = [
        torch.randn(B, 1, H, W, requires_grad=True),
        torch.randn(B, 1, H, W, requires_grad=True),
        torch.randn(B, 1, H, W, requires_grad=True),
    ]
    target = (torch.rand(B, 1, H, W) > 0.8).float()

    loss, loss_dict = criterion(preds, target)
    print(f"  Total loss:  {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Check gradient flows
    loss.backward()
    assert preds[0].grad is not None, "Gradients should flow to predictions"

    # Verify distance weighting: loss should be higher for near-field errors
    print("\n  Distance weighting verification:")
    print(f"  Weight map shape: {criterion.loss_fn.weight_map.shape}")
    wm = criterion.loss_fn.weight_map.squeeze()
    center = wm[H // 2, W // 2]
    corner = wm[0, 0]
    print(f"  Center weight (near ego): {center:.3f}")
    print(f"  Corner weight (far):      {corner:.3f}")
    assert center > corner, "Center (near ego) should have higher weight"
    print("  ✓ PASSED\n")


def test_metrics():
    """Test evaluation metrics."""
    print("=" * 60)
    print("TEST 6: Evaluation Metrics")
    print("=" * 60)

    from utils.metrics import OccupancyMetrics

    metrics = OccupancyMetrics()

    # Perfect prediction
    pred = torch.ones(4, 200, 200) * 0.9
    target = torch.ones(4, 200, 200)
    metrics.update(pred, target)
    results = metrics.compute()
    print(f"  Perfect match IoU: {results['occupancy_iou']:.4f}")
    assert results["occupancy_iou"] > 0.99

    # Reset and test with noise
    metrics.reset()
    pred = torch.rand(4, 200, 200)
    target = (torch.rand(4, 200, 200) > 0.8).float()
    metrics.update(pred, target)
    results = metrics.compute()
    print(f"  Random pred IoU:   {results['occupancy_iou']:.4f}")
    print(f"  Random pred DWE:   {results['distance_weighted_error']:.4f}")

    # Check per-distance bins exist
    for key in results:
        if "iou_" in key:
            print(f"  {key}: {results[key]:.4f}")

    print("  ✓ PASSED\n")


def test_full_model():
    """Test full BEV-Occ model end-to-end."""
    print("=" * 60)
    print("TEST 7: Full Model End-to-End")
    print("=" * 60)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "configs", "bevocc_efficientb4.py")
    )
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.config

    from models.bevocc import BEVOcc

    # Use non-pretrained for speed
    # Override config to use smaller settings for test
    cfg["model"]["use_temporal"] = False  # Skip temporal to save memory
    model = BEVOcc(cfg)
    model.eval()

    B, N = 1, 2  # Fewer cameras for memory
    img_h, img_w = 128, 352  # Smaller images

    batch = {
        "images": torch.randn(B, N, 3, img_h, img_w),
        "intrinsics": torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone(),
        "extrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone(),
        "ego_motion": torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone(),
    }

    # Set reasonable intrinsics
    batch["intrinsics"][:, :, 0, 0] = 1266.0
    batch["intrinsics"][:, :, 1, 1] = 1266.0
    batch["intrinsics"][:, :, 0, 2] = 352.0
    batch["intrinsics"][:, :, 1, 2] = 128.0

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters: {params:.1f}M")

    # Forward
    t0 = time.time()
    with torch.no_grad():
        output = model(batch)
    elapsed = time.time() - t0

    print(f"  Forward time: {elapsed:.2f}s")
    print(f"  Predictions:  {len(output['predictions'])} scales")
    for i, p in enumerate(output["predictions"]):
        print(f"    Scale {i}: {p.shape}")
    print(f"  Hidden state: {output['hidden'].shape}")
    print(f"  Depth dist:   {output['depth'].shape}")

    # Check output values are reasonable
    pred = torch.sigmoid(output["predictions"][0])
    print(f"  Pred range:   [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    # Backward
    target = torch.zeros(B, 1, 200, 200)
    from utils.losses import MultiScaleLoss
    criterion = MultiScaleLoss()
    loss, _ = criterion(output["predictions"], target)
    loss.backward()
    print(f"  Loss:         {loss.item():.4f}")
    print(f"  Gradient check: OK (backward completed)")

    # Inference mode
    model.eval()
    with torch.no_grad():
        binary, prob, hidden = model.predict(batch)
    print(f"  Predict output: binary={binary.shape}, prob={prob.shape}")

    print("  ✓ PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("BEV-Occ Sanity Check")
    print("Verifying all components work correctly")
    print("=" * 60 + "\n")

    tests = [
        ("Backbone + DepthNet", test_backbone),
        ("Lift-Splat Transform", test_lift_splat),
        ("Temporal Fusion", test_temporal_fusion),
        ("BEV Encoder + Head", test_bev_encoder),
        ("Loss Functions", test_losses),
        ("Evaluation Metrics", test_metrics),
        ("Full Model E2E", test_full_model),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("All tests passed! Model is ready for training.")
    else:
        print(f"{failed} test(s) failed. Please fix before proceeding.")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
