#!/usr/bin/env python3
# scripts/generate_gt.py
"""
Generate BEV occupancy ground truth from nuScenes LiDAR point clouds.

For each sample, aggregates multi-sweep LiDAR points, projects them
to the ego frame, and creates a 2D occupancy grid.

Usage:
    python scripts/generate_gt.py --dataroot data/nuscenes --version v1.0-trainval
    python scripts/generate_gt.py --dataroot data/nuscenes --version v1.0-mini
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


def get_transform(translation, rotation):
    """Create 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


def aggregate_lidar_sweeps(nusc, sample, num_sweeps=10):
    """
    Aggregate LiDAR points from multiple sweeps into the current ego frame.
    
    More sweeps = denser point cloud = better ground truth.
    """
    # Get current LIDAR sample data
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)

    # Current ego pose
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    T_ego_to_global = get_transform(ego_pose["translation"], ego_pose["rotation"])
    T_global_to_ego = np.linalg.inv(T_ego_to_global)

    # Current sensor calibration
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    T_sensor_to_ego = get_transform(calib["translation"], calib["rotation"])

    all_points = []

    # Current sweep
    pc_path = os.path.join(nusc.dataroot, sd["filename"])
    pc = LidarPointCloud.from_file(pc_path)
    points = pc.points[:3, :].T  # (N, 3) in sensor frame

    # Sensor → ego frame
    points_h = np.hstack([points, np.ones((len(points), 1))])
    points_ego = (T_sensor_to_ego @ points_h.T).T[:, :3]
    all_points.append(points_ego)

    # Previous sweeps
    current_sd = sd
    for _ in range(num_sweeps - 1):
        if current_sd["prev"] == "":
            break
        current_sd = nusc.get("sample_data", current_sd["prev"])

        # Load point cloud
        pc_path = os.path.join(nusc.dataroot, current_sd["filename"])
        pc = LidarPointCloud.from_file(pc_path)
        points = pc.points[:3, :].T

        # Sweep sensor → sweep ego
        sweep_calib = nusc.get(
            "calibrated_sensor", current_sd["calibrated_sensor_token"]
        )
        T_sweep_sensor_to_ego = get_transform(
            sweep_calib["translation"], sweep_calib["rotation"]
        )

        # Sweep ego → global
        sweep_ego_pose = nusc.get("ego_pose", current_sd["ego_pose_token"])
        T_sweep_ego_to_global = get_transform(
            sweep_ego_pose["translation"], sweep_ego_pose["rotation"]
        )

        # Full chain: sweep_sensor → sweep_ego → global → current_ego
        T_full = T_global_to_ego @ T_sweep_ego_to_global @ T_sweep_sensor_to_ego
        points_h = np.hstack([points, np.ones((len(points), 1))])
        points_ego = (T_full @ points_h.T).T[:, :3]
        all_points.append(points_ego)

    return np.concatenate(all_points, axis=0)


def create_occupancy_grid(points, x_bound=(-50.0, 50.0, 0.5),
                          y_bound=(-50.0, 50.0, 0.5),
                          z_min=-3.0, z_max=3.0,
                          min_points=1):
    """
    Create a 2D BEV occupancy grid from 3D LiDAR points.
    
    Args:
        points: (N, 3) points in ego frame
        x_bound, y_bound: grid bounds [min, max, resolution]
        z_min, z_max: height filter (ignore ground below z_min)
        min_points: minimum points in a cell to mark as occupied
    
    Returns:
        grid: (H, W) binary occupancy grid
    """
    nx = int((x_bound[1] - x_bound[0]) / x_bound[2])  # 200
    ny = int((y_bound[1] - y_bound[0]) / y_bound[2])  # 200

    # Filter by height (remove ground returns and sky)
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    points = points[mask]

    # Filter by BEV bounds
    mask = (
        (points[:, 0] >= x_bound[0]) & (points[:, 0] < x_bound[1]) &
        (points[:, 1] >= y_bound[0]) & (points[:, 1] < y_bound[1])
    )
    points = points[mask]

    if len(points) == 0:
        return np.zeros((ny, nx), dtype=np.float32)

    # Compute grid indices
    x_idx = ((points[:, 0] - x_bound[0]) / x_bound[2]).astype(np.int32)
    y_idx = ((points[:, 1] - y_bound[0]) / y_bound[2]).astype(np.int32)

    # Clamp
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)

    # Count points per cell
    grid = np.zeros((ny, nx), dtype=np.float32)
    np.add.at(grid, (y_idx, x_idx), 1)

    # Binary: occupied if >= min_points
    grid = (grid >= min_points).astype(np.float32)

    return grid


def main():
    parser = argparse.ArgumentParser(description="Generate BEV GT from LiDAR")
    parser.add_argument("--dataroot", type=str, default="data/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-sweeps", type=int, default=10)
    parser.add_argument("--x-bound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    parser.add_argument("--y-bound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.dataroot, "bev_gt")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    print(f"Generating BEV GT with {args.num_sweeps} sweeps...")
    print(f"Grid: x={args.x_bound}, y={args.y_bound}")
    print(f"Output: {args.output_dir}")

    for sample in tqdm(nusc.sample, desc="Generating GT"):
        token = sample["token"]
        out_path = os.path.join(args.output_dir, f"{token}.npy")

        if os.path.exists(out_path):
            continue

        # Aggregate LiDAR sweeps
        points = aggregate_lidar_sweeps(nusc, sample, num_sweeps=args.num_sweeps)

        # Create occupancy grid
        grid = create_occupancy_grid(
            points,
            x_bound=tuple(args.x_bound),
            y_bound=tuple(args.y_bound),
        )

        # Save
        np.save(out_path, grid)

    print(f"Done! Generated {len(nusc.sample)} ground truth grids.")

    # Print statistics
    grids = []
    for f in os.listdir(args.output_dir):
        if f.endswith(".npy"):
            grids.append(np.load(os.path.join(args.output_dir, f)))
            if len(grids) >= 100:
                break

    if grids:
        occ_rates = [g.mean() for g in grids]
        print(f"\nGT Statistics (first {len(grids)} samples):")
        print(f"  Mean occupancy rate: {np.mean(occ_rates):.4f}")
        print(f"  Min occupancy rate:  {np.min(occ_rates):.4f}")
        print(f"  Max occupancy rate:  {np.max(occ_rates):.4f}")


if __name__ == "__main__":
    main()
