"""
MediaPipe Joints Dataset with Failed Detection Filtering

Now SKIPS samples where MediaPipe cannot detect a pose, rather than
including them with all-zero joint coordinates.
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import mediapipe as mp
import random
import math


class MediaPipeJointsDataset(Dataset):
    """
    Dataset that extracts MediaPipe pose joints from images.
    
    Automatically filters out samples where pose detection fails.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        use_kinect_depth: bool = False,
        include_visibility: bool = False,
        cache_joints: bool = True,
        cache_dir: Optional[str] = None,
        augment: bool = True,
        # Augmentation parameters
        joint_noise_prob: float = 0.5,
        joint_noise_std: float = 0.01,
        horizontal_flip_prob: float = 0.5,
        # Azure Kinect calibration
        K: Optional[np.ndarray] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        # Detection filtering
        skip_failed_detections: bool = True,  # NEW: Skip samples with no pose
        min_visibility: float = 0.5,  # NEW: Minimum joint visibility
    ):
        """
        Args:
            data_dir: Path to directory containing .jpg, .npy, and .txt files
            use_kinect_depth: If True, use Azure Kinect depth
            skip_failed_detections: If True, skip samples where pose detection fails
            min_visibility: Minimum average visibility score (0-1) to keep sample
            ... (other args same as before)
        """
        self.data_dir = data_dir
        self.use_kinect_depth = use_kinect_depth
        self.include_visibility = include_visibility
        self.cache_joints = cache_joints
        self.augment = augment
        self.skip_failed_detections = skip_failed_detections
        self.min_visibility = min_visibility
        
        # Augmentation settings
        self.joint_noise_prob = joint_noise_prob
        self.joint_noise_std = joint_noise_std
        self.horizontal_flip_prob = horizontal_flip_prob
        
        # Azure Kinect calibration
        if K is not None:
            self.fx = float(K[0, 0])
            self.fy = float(K[1, 1])
            self.cx = float(K[0, 2])
            self.cy = float(K[1, 2])
        else:
            self.fx = fx if fx is not None else 1078.0
            self.fy = fy if fy is not None else 1078.0
            self.cx = cx if cx is not None else 960.0
            self.cy = cy if cy is not None else 540.0
        
        # Setup cache directory
        if cache_dir is None:
            depth_suffix = "_kinect" if use_kinect_depth else "_mediapipe"
            cache_dir = os.path.join(data_dir, f'joints_cache{depth_suffix}')
        self.cache_dir = cache_dir
        if cache_joints:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Find all potential samples
        potential_samples = []
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg'):
                    base_name = filename[:-4]
                    sample = {
                        'base_name': base_name,
                        'image_path': os.path.join(data_dir, filename),
                        'depth_path': os.path.join(data_dir, base_name + '.npy'),
                        'label_path': os.path.join(data_dir, base_name + '.txt'),
                        'cache_path': os.path.join(cache_dir, base_name + '_joints.npy')
                    }
                    # Check depth file exists if using Kinect
                    if use_kinect_depth:
                        if os.path.exists(sample['depth_path']):
                            potential_samples.append(sample)
                    else:
                        potential_samples.append(sample)
        
        # Filter out samples with failed pose detection
        if skip_failed_detections:
            self.samples = self._filter_valid_samples(potential_samples)
            print(f"Filtered {len(potential_samples) - len(self.samples)} samples with failed pose detection")
        else:
            self.samples = potential_samples
        
        feature_dim = 99  # 33 joints × 3 coords
        if include_visibility:
            feature_dim = 132
        
        depth_mode = "Azure Kinect" if use_kinect_depth else "MediaPipe"
        aug_status = "ON" if augment else "OFF"
        print(f"MediaPipe Joints Dataset: {len(self.samples)} samples, augmentation {aug_status}")
        print(f"  - Depth mode: {depth_mode}")
        print(f"  - Feature dimension: {feature_dim}")
        print(f"  - Caching: {cache_joints}")
        print(f"  - Skip failed detections: {skip_failed_detections}")
        if use_kinect_depth:
            calib_source = "K matrix" if K is not None else "individual params"
            print(f"  - Calibration: {calib_source}")
            print(f"  - Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
    
    def _filter_valid_samples(self, potential_samples: List[dict]) -> List[dict]:
        """
        Filter samples to only keep those where pose detection succeeds.
        
        Returns:
            List of valid samples (with detected poses)
        """
        print(f"Checking {len(potential_samples)} samples for valid pose detections...")
        from tqdm import tqdm
        
        valid_samples = []
        
        for sample in tqdm(potential_samples, desc="Filtering samples"):
            # Try to load from cache first
            if self.cache_joints and os.path.exists(sample['cache_path']):
                joints = np.load(sample['cache_path'])
                # Check if it's all zeros (failed detection)
                if np.allclose(joints, 0.0):
                    continue  # Skip this sample
                valid_samples.append(sample)
            else:
                # Extract joints and check if detection succeeded
                try:
                    if self.use_kinect_depth:
                        joints = self._extract_joints_kinect(
                            sample['image_path'],
                            sample['depth_path'],
                            check_only=True
                        )
                    else:
                        joints = self._extract_joints_mediapipe(
                            sample['image_path'],
                            check_only=True
                        )
                    
                    # If joints are not all zeros, pose was detected
                    if joints is not None and not np.allclose(joints, 0.0):
                        # Check average visibility if applicable
                        if self.include_visibility:
                            vis_scores = joints[3::4]  # Every 4th value is visibility
                            avg_vis = np.mean(vis_scores)
                            if avg_vis >= self.min_visibility:
                                valid_samples.append(sample)
                                # Cache it
                                if self.cache_joints:
                                    np.save(sample['cache_path'], joints)
                        else:
                            valid_samples.append(sample)
                            # Cache it
                            if self.cache_joints:
                                np.save(sample['cache_path'], joints)
                except Exception as e:
                    print(f"Error processing {sample['image_path']}: {e}")
                    continue
        
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Returns joint features and labels"""
        sample = self.samples[idx]
        
        # Load from cache (we know it's valid if we got here)
        if self.cache_joints and os.path.exists(sample['cache_path']):
            joint_features = np.load(sample['cache_path'])
        else:
            # Extract joints
            if self.use_kinect_depth:
                joint_features = self._extract_joints_kinect(
                    sample['image_path'], 
                    sample['depth_path']
                )
            else:
                joint_features = self._extract_joints_mediapipe(sample['image_path'])
            
            # Cache it
            if self.cache_joints:
                np.save(sample['cache_path'], joint_features)
        
        # Load labels
        label_dict = self._load_label(sample['label_path'])
        
        # Convert to torch tensors
        joints = torch.from_numpy(joint_features).float()
        is_pointing = label_dict['is_pointing'] == 1
        direction = torch.from_numpy(label_dict['pointing_vector']).float()
        
        # Apply augmentation
        if self.augment:
            joints, direction = self._apply_augmentation(joints, direction, is_pointing)
        
        label_dict['pointing_vector'] = direction
        
        return joints, label_dict
    
    def _extract_joints_mediapipe(self, image_path: str, check_only: bool = False) -> Optional[np.ndarray]:
        """Extract MediaPipe world landmarks"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks and results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            joint_coords = []
            for lm in landmarks:
                joint_coords.extend([lm.x, lm.y, lm.z])
                if self.include_visibility:
                    vis = results.pose_landmarks.landmark[len(joint_coords)//3 - 1].visibility
                    joint_coords.append(vis)
            
            return np.array(joint_coords, dtype=np.float32)
        else:
            # Detection failed
            if not check_only:
                print(f"Warning: No pose detected in {image_path}")
            return None  # Return None instead of zeros
    
    def _extract_joints_kinect(self, image_path: str, depth_path: str, check_only: bool = False) -> Optional[np.ndarray]:
        """Extract joints using MediaPipe 2D + Azure Kinect depth"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth_mm = np.load(depth_path)
        
        H, W = image_rgb.shape[:2]
        
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            joint_coords = []
            for lm in landmarks:
                x_px = int(lm.x * W)
                y_px = int(lm.y * H)
                
                x_px = max(0, min(W - 1, x_px))
                y_px = max(0, min(H - 1, y_px))
                
                depth_value = depth_mm[y_px, x_px]
                
                if depth_value > 0:
                    z_mm = float(depth_value)
                    x_mm = (x_px - self.cx) * z_mm / self.fx
                    y_mm = (y_px - self.cy) * z_mm / self.fy
                    
                    x_m = x_mm / 1000.0
                    y_m = y_mm / 1000.0
                    z_m = z_mm / 1000.0
                else:
                    x_m, y_m, z_m = 0.0, 0.0, 0.0
                
                joint_coords.extend([x_m, y_m, z_m])
                
                if self.include_visibility:
                    joint_coords.append(lm.visibility)
            
            return np.array(joint_coords, dtype=np.float32)
        else:
            # Detection failed
            if not check_only:
                print(f"Warning: No pose detected in {image_path}")
            return None
    
    def _apply_augmentation(self, joints: torch.Tensor, direction: torch.Tensor, is_pointing: bool):
        """Apply augmentations to joints and direction"""
        num_features_per_joint = 4 if self.include_visibility else 3
        joints_reshaped = joints.view(33, num_features_per_joint)
        
        # Add noise
        if random.random() < self.joint_noise_prob:
            noise = torch.randn_like(joints_reshaped[:, :3]) * self.joint_noise_std
            joints_reshaped[:, :3] = joints_reshaped[:, :3] + noise
        
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            joints_reshaped[:, 0] = -joints_reshaped[:, 0]
            
            if is_pointing:
                direction = direction.clone()
                direction[0] = -direction[0]
        
        joints = joints_reshaped.flatten()
        
        return joints, direction
    
    def _load_label(self, label_path: str) -> dict:
        """Load label from .txt file"""
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        label = int(lines[0].strip())
        
        result = {'is_pointing': label}
        
        if label == 1 and len(lines) >= 7:
            pointing_vector = np.array([
                float(lines[4].strip()),
                float(lines[5].strip()),
                float(lines[6].strip())
            ])
            result['pointing_vector'] = pointing_vector
        else:
            result['pointing_vector'] = np.array([0.0, 0.0, 0.0])
        
        return result
    
    def precompute_all_joints(self):
        """Precompute and cache all joints"""
        print(f"\nPrecomputing joints for {len(self.samples)} samples...")
        from tqdm import tqdm
        
        for idx in tqdm(range(len(self.samples))):
            sample = self.samples[idx]
            if not os.path.exists(sample['cache_path']):
                if self.use_kinect_depth:
                    joint_features = self._extract_joints_kinect(
                        sample['image_path'],
                        sample['depth_path']
                    )
                else:
                    joint_features = self._extract_joints_mediapipe(sample['image_path'])
                
                if joint_features is not None:
                    np.save(sample['cache_path'], joint_features)
        
        print("✓ Joints precomputed and cached!")


# Convenience functions
def create_train_dataset(data_dir: str, use_kinect_depth: bool = False, K: Optional[np.ndarray] = None, **kwargs):
    """Create training dataset WITH augmentation"""
    return MediaPipeJointsDataset(
        data_dir=data_dir,
        use_kinect_depth=use_kinect_depth,
        K=K,
        augment=True,
        include_visibility=False,
        cache_joints=True,
        skip_failed_detections=True,  # Default: skip failed detections
        **kwargs
    )


def create_val_dataset(data_dir: str, use_kinect_depth: bool = False, K: Optional[np.ndarray] = None, **kwargs):
    """Create validation dataset WITHOUT augmentation"""
    return MediaPipeJointsDataset(
        data_dir=data_dir,
        use_kinect_depth=use_kinect_depth,
        K=K,
        augment=False,
        include_visibility=False,
        cache_joints=True,
        skip_failed_detections=True,  # Default: skip failed detections
        **kwargs
    )


if __name__ == "__main__":
    print("="*70)
    print("TESTING MEDIAPIPE DATASET WITH DETECTION FILTERING")
    print("="*70)
    
    K = np.array([[919.76178, 0, 962.6875],
                  [0, 919.8909, 550.9944],
                  [0, 0, 1]], dtype=np.float64)
    
    # This will automatically filter out samples with no pose
    train_dataset = create_train_dataset("./split_data/train", use_kinect_depth=True, K=K)
    val_dataset = create_val_dataset("./split_data/val", use_kinect_depth=True, K=K)
    
    print(f"\n✓ Valid samples: train={len(train_dataset)}, val={len(val_dataset)}")