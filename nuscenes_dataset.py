# data/nuscenes_dataset.py
"""
nuScenes dataset for BEV occupancy training.

Loads multi-camera images, camera calibration, ego-motion,
and BEV ground truth generated from LiDAR point clouds.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pyquaternion import Quaternion

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
except ImportError:
    print("WARNING: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")


class NuScenesBEVDataset(Dataset):
    """
    nuScenes dataset for BEV occupancy prediction.
    
    Each sample provides:
    - 6 camera images (resized)
    - Camera intrinsics and extrinsics
    - BEV occupancy ground truth (from LiDAR)
    - Ego-motion between consecutive frames (for temporal fusion)
    """

    CAMERAS = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ]

    def __init__(self, nusc, split="train", config=None):
        """
        Args:
            nusc: NuScenes instance
            split: "train" or "val"
            config: data config dict
        """
        self.nusc = nusc
        self.config = config or {}
        self.input_size = self.config.get("input_size", (256, 704))
        self.gt_dir = self.config.get("gt_dir", "data/nuscenes/bev_gt")

        # Get scene splits
        splits = create_splits_scenes()
        scene_names = splits[split]

        # Collect all sample tokens for the split
        self.samples = []
        for scene in self.nusc.scene:
            if scene["name"] in scene_names:
                sample_token = scene["first_sample_token"]
                while sample_token:
                    self.samples.append(sample_token)
                    sample = self.nusc.get("sample", sample_token)
                    sample_token = sample["next"]

        print(f"NuScenesBEVDataset: {split} split with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_token = self.samples[idx]
        sample = self.nusc.get("sample", sample_token)

        images = []
        intrinsics_list = []
        extrinsics_list = []

        for cam_name in self.CAMERAS:
            cam_data = self.nusc.get("sample_data", sample["data"][cam_name])

            # Load and resize image
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)

            # Normalize to [0, 1] then ImageNet normalization
            img_np = np.array(img).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np - mean) / std
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            images.append(img_tensor)

            # Camera intrinsics
            cam_calib = self.nusc.get(
                "calibrated_sensor", cam_data["calibrated_sensor_token"]
            )
            K = np.array(cam_calib["camera_intrinsic"])  # 3x3

            # Adjust intrinsics for resize
            scale_x = self.input_size[1] / orig_w
            scale_y = self.input_size[0] / orig_h
            K[0, :] *= scale_x
            K[1, :] *= scale_y
            intrinsics_list.append(torch.from_numpy(K).float())

            # Extrinsics: camera → ego frame
            # Translation and rotation from calibrated_sensor
            T_cam_to_ego = self._get_transform(
                cam_calib["translation"], cam_calib["rotation"]
            )

            # Ego → global (for the camera's timestamp)
            ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
            T_ego_to_global = self._get_transform(
                ego_pose["translation"], ego_pose["rotation"]
            )

            # We want camera → ego (for projection)
            # The full chain is camera → ego → global, but for BEV
            # we work in ego frame, so just camera → ego
            extrinsics_list.append(torch.from_numpy(T_cam_to_ego).float())

        images = torch.stack(images)          # (N, 3, H, W)
        intrinsics = torch.stack(intrinsics_list)  # (N, 3, 3)
        extrinsics = torch.stack(extrinsics_list)  # (N, 4, 4)

        # Load BEV ground truth
        gt_path = os.path.join(self.gt_dir, f"{sample_token}.npy")
        if os.path.exists(gt_path):
            bev_gt = torch.from_numpy(np.load(gt_path)).float()
        else:
            # Placeholder if GT not generated yet
            bev_gt = torch.zeros(200, 200)

        # Ego motion (current → previous frame, for temporal fusion)
        ego_motion = self._get_ego_motion(sample)

        return {
            "images": images,              # (N, 3, H, W)
            "intrinsics": intrinsics,      # (N, 3, 3)
            "extrinsics": extrinsics,      # (N, 4, 4)
            "bev_gt": bev_gt,              # (H_bev, W_bev)
            "ego_motion": ego_motion,      # (4, 4)
            "sample_token": sample_token,
        }

    def _get_transform(self, translation, rotation):
        """Create 4x4 transformation matrix from translation + quaternion."""
        T = np.eye(4)
        T[:3, :3] = Quaternion(rotation).rotation_matrix
        T[:3, 3] = translation
        return T

    def _get_ego_motion(self, sample):
        """
        Compute ego-motion from previous sample to current sample.
        Returns 4x4 transformation matrix.
        """
        if sample["prev"] == "":
            return torch.eye(4).float()

        # Current ego pose
        curr_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        curr_pose = self.nusc.get("ego_pose", curr_sd["ego_pose_token"])
        T_curr = self._get_transform(curr_pose["translation"], curr_pose["rotation"])

        # Previous ego pose
        prev_sample = self.nusc.get("sample", sample["prev"])
        prev_sd = self.nusc.get("sample_data", prev_sample["data"]["LIDAR_TOP"])
        prev_pose = self.nusc.get("ego_pose", prev_sd["ego_pose_token"])
        T_prev = self._get_transform(prev_pose["translation"], prev_pose["rotation"])

        # Ego motion: transform from prev ego frame to curr ego frame
        # p_curr = T_curr_inv @ T_prev @ p_prev
        T_motion = np.linalg.inv(T_curr) @ T_prev

        return torch.from_numpy(T_motion).float()


def collate_fn(batch):
    """Custom collate function for the dataloader."""
    return {
        "images": torch.stack([b["images"] for b in batch]),
        "intrinsics": torch.stack([b["intrinsics"] for b in batch]),
        "extrinsics": torch.stack([b["extrinsics"] for b in batch]),
        "bev_gt": torch.stack([b["bev_gt"] for b in batch]),
        "ego_motion": torch.stack([b["ego_motion"] for b in batch]),
        "sample_tokens": [b["sample_token"] for b in batch],
    }
