# BEV-Occ: Bird's-Eye-View 2D Occupancy Grid from Camera Images

## Overview

**BEV-Occ** transforms multi-camera front-view images into a 2D top-down occupancy grid for autonomous driving. Our approach builds on the Lift-Splat-Shoot (LSS) paradigm with key enhancements:

1. **EfficientNet-B4 Backbone** — Strong feature extraction with manageable compute
2. **Lift-Splat-Shoot BEV Transform** — Predicts depth distributions per pixel, "lifts" 2D features into 3D, then "splats" onto BEV plane
3. **Temporal BEV Fusion** — Warps previous BEV features using ego-motion for temporal consistency
4. **Distance-Weighted Loss** — Custom loss that penalizes near-field errors more heavily, directly aligned with evaluation metrics
5. **Multi-Scale Occupancy Head** — Coarse-to-fine prediction with deep supervision

## Architecture

```
Multi-Camera Images (6 views)
        │
        ▼
┌─────────────────────┐
│  EfficientNet-B4     │  (ImageNet pretrained)
│  Feature Extractor   │
└─────────┬───────────┘
          │ C×H/8×W/8 features per view
          ▼
┌─────────────────────┐
│  Depth Distribution  │  Predict D depth bins per pixel
│  Network (DepthNet)  │
└─────────┬───────────┘
          │ C×D×H/8×W/8 frustum features
          ▼
┌─────────────────────┐
│  Voxel Pooling       │  Splat frustum → 3D voxel grid
│  (Lift-Splat)        │  Collapse Z → BEV: C×X×Y
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Temporal Fusion     │  Warp prev BEV via ego-motion
│  (GRU + Warp)        │  Fuse with current BEV
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  BEV Encoder         │  ResNet-18 neck on BEV plane
│  (Multi-Scale)       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Occupancy Head      │  Sigmoid → 200×200 grid
│  + Distance Loss     │  (0.5m resolution, 100m×100m)
└─────────────────────┘
```

## Setup

```bash
# Clone
git clone https://github.com/dussa-harshitha/bev-occupancy.git
cd bev-occupancy

# Environment
conda create -n bevocc python=3.9 -y
conda activate bevocc
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install nuscenes-devkit efficientnet_pytorch opencv-python tensorboard tqdm

# Download nuScenes
# Place dataset at data/nuscenes/ with structure:
# data/nuscenes/
#   ├── maps/
#   ├── samples/
#   ├── sweeps/
#   ├── v1.0-trainval/
#   └── v1.0-mini/

# Generate BEV ground truth from LiDAR
python scripts/generate_gt.py --dataroot data/nuscenes --version v1.0-trainval

# Train
python train.py --config configs/bevocc_efficientb4.py

# Evaluate
python evaluate.py --config configs/bevocc_efficientb4.py --checkpoint checkpoints/best.pth

# Inference / Visualize
python inference.py --config configs/bevocc_efficientb4.py --checkpoint checkpoints/best.pth --sample-token <token>
```

## Results (nuScenes val)

| Model | Occupancy IoU | Distance-Weighted Error | FPS |
|-------|:---:|:---:|:---:|
| IPM Baseline | 28.3 | 0.412 | 45 |
| LSS (reproduced) | 38.7 | 0.298 | 22 |
| **BEV-Occ (Ours)** | **42.1** | **0.251** | **18** |

<img width="1926" height="997" alt="image" src="https://github.com/user-attachments/assets/d0552043-3329-4579-a836-fa942b60c989" />
<img width="2019" height="612" alt="image" src="https://github.com/user-attachments/assets/5c76ef0d-87aa-4aa3-8c03-19558bf13312" />


## Key Novelties

- **Metric-Aligned Training**: Distance-weighted BCE loss directly optimizes the evaluation criterion
- **Temporal Fusion**: Ego-motion-compensated BEV warping with GRU fusion improves occluded regions
- **Multi-Sweep GT**: 10-sweep LiDAR aggregation for denser supervision signal
- **Height-Aware Pooling**: Learned height bins separate ground, vehicle, and tall obstacle features

## Team

Team Name: Immobile Creatives
Team Members: Jahnavi Sri Sai Akveti, Nakshatra Vinoth, Harshitha Dussa, Jonath Jimmi

## License

MIT
