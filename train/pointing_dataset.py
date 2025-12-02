import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import shutil
import random
from pathlib import Path
import einops
import torchvision.transforms.functional as TF
import math

class PointingDataset(Dataset):
    """
    Dataset for 4D (RGB-D) pointing gesture recognition.
    
    Data format per sample:
    - RGB image: {timestamp}.jpg (1920x1080x3, BGR uint8)
    - Depth image: {timestamp}.npy (1080x1920, uint16, millimeters)
    - Label: {timestamp}.txt
        - Line 1: label (0 or 1)
        - Lines 2-7 (if label==1): 6 floats (wrist_x, wrist_y, wrist_z, dir_x, dir_y, dir_z)
    """
    
    def __init__(self, 
        data_dir: str, 
        transform: Optional[callable] = None, 
        use_depth: bool = False,
<<<<<<< HEAD
        augment: bool = True,
=======
        augment: bool = False,
>>>>>>> 85dfa7a61e7f0e9b1e04645a915096120e90aea6
        color_jitter_prob: float = 0.8,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        gaussian_blur_prob: float = 0.3,
        gaussian_noise_prob: float = 0.3,
        noise_std: float = 0.05,
        # Geometric augmentations (with vector correction)
        horizontal_flip_prob: float = 0.5,
        rotation_prob: float = 0.3,
        max_rotation_degrees: float = 15.0,
        normalize: bool = True,
        ):
        """
        Args:
            data_dir: Path to directory containing .jpg, .npy, and .txt files
            transform: Optional transform to apply to the image
        """
        self.data_dir = data_dir
        self.transform = transform
        self.use_depth = use_depth
        self.augment = augment
        self.normalize = normalize
        # Add in __init__:
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # Augmentation settings
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_std = noise_std
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_prob = rotation_prob
        self.max_rotation_degrees = max_rotation_degrees
        # find all .jpg files (each represents a complete sample)
        self.samples = []
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg'):
                    base_name = filename[:-4]  # remove .jpg extension
                    sample = {
                        'base_name': base_name,
                        'image_path': os.path.join(data_dir, filename),
                        'depth_path': os.path.join(data_dir, base_name + '.npy'),
                        'label_path': os.path.join(data_dir, base_name + '.txt')
                    }
                    self.samples.append(sample)
        
        print(f"Found {len(self.samples)} samples in {data_dir}, augmentations: {augment}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        
        # Load RGB image
        bgr_img = cv2.imread(sample['image_path'])
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
<<<<<<< HEAD
        rgb_img = rgb_img.astype(np.float32) / 255.0
=======
>>>>>>> 85dfa7a61e7f0e9b1e04645a915096120e90aea6
        rgb_img = einops.rearrange(rgb_img, 'h w c -> c h w')
        
        # Load depth if needed
        if self.use_depth:
            depth_img = np.load(sample['depth_path'])
            depth_img = depth_img.astype(np.float32) / 1000.0
            depth_img = np.clip(depth_img, 0, 10.0)
            depth_img = np.expand_dims(depth_img, axis=0)
            fin_img = np.concatenate([rgb_img, depth_img], axis=0)
        else:
            fin_img = rgb_img
        
        # Apply transform if provided
        image = torch.from_numpy(fin_img).float()
        if self.transform:
            image = self.transform(image)
        else:
            image = image / 255.0  

        # Load labels
        label_dict = self._load_label(sample['label_path'])
        
        # Convert to torch tensors
<<<<<<< HEAD
        image = torch.from_numpy(fin_img).float()
=======
>>>>>>> 85dfa7a61e7f0e9b1e04645a915096120e90aea6
        is_pointing = label_dict['is_pointing'] == 1
        direction = torch.from_numpy(label_dict['pointing_vector']).float()
        
        # Augment (image AND direction)
        if self.augment:
            image, direction = self._apply_augmentation(image, direction, is_pointing)
        
        # Normalize
        if self.normalize:
            image = (image - self.mean) / self.std

        # Update label dict with augmented direction
        label_dict['pointing_vector'] = direction
        if label_dict['wrist_coords'] is not None:
            label_dict['wrist_coords'] = torch.from_numpy(label_dict['wrist_coords']).float()

        return image, label_dict  # ✅ Return augmented image


    def _apply_augmentation(self, image: torch.Tensor, direction: torch.Tensor, is_pointing: bool):
        """
        Apply augmentations to image and direction.
        Called within __getitem__, so DataLoader parallelizes this!
        """
        
        # =====================================================================
        # SAFE AUGMENTATIONS (no vector correction needed)
        # =====================================================================
        
        # Color jitter
        if random.random() < self.color_jitter_prob:
            # Random brightness
            brightness_factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = TF.adjust_brightness(image, brightness_factor)
            
            # Random contrast
            contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)
            image = TF.adjust_contrast(image, contrast_factor)
            
            # Random saturation
            saturation_factor = 1 + random.uniform(-self.saturation, self.saturation)
            image = TF.adjust_saturation(image, saturation_factor)
        
        # Gaussian blur
        if random.random() < self.gaussian_blur_prob:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 2.0)
            image = TF.gaussian_blur(image, kernel_size, [sigma, sigma])
        
        # Gaussian noise
        if random.random() < self.gaussian_noise_prob:
            noise = torch.randn_like(image) * self.noise_std
            image = torch.clamp(image + noise, 0, 1)
        
        # =====================================================================
        # GEOMETRIC AUGMENTATIONS (must transform direction too!)
        # =====================================================================
        
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            if is_pointing:
                direction = direction.clone()
                direction[0] = -direction[0]  # Flip X component
        
        # Rotation
        # if random.random() < self.rotation_prob:
        #     angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
        #     image = TF.rotate(image, angle)
        #     if is_pointing:
        #         direction = self._rotate_direction_z(direction, angle)
        
        return image, direction
    
    def _rotate_direction_z(self, direction: torch.Tensor, angle_degrees: float) -> torch.Tensor:
        """Rotate direction vector around Z-axis (camera optical axis)"""
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        x, y, z = direction[0].item(), direction[1].item(), direction[2].item()
        
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        new_z = z
        
        return torch.tensor([new_x, new_y, new_z], dtype=direction.dtype)

    def _load_label(self, label_path: str) -> dict:
        """Load label from .txt file"""
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        label = int(lines[0].strip())
        
        result = {
            'is_pointing': label
        }
        
        if label == 1 and len(lines) >= 7:
            # Parse wrist coordinates and pointing vector
            wrist_coords = np.array([
                float(lines[1].strip()),
                float(lines[2].strip()),
                float(lines[3].strip())
            ])
            pointing_vector = np.array([
                float(lines[4].strip()),
                float(lines[5].strip()),
                float(lines[6].strip())
            ])
            result['wrist_coords'] = wrist_coords
            result['pointing_vector'] = pointing_vector
        else:
            result['wrist_coords'] = np.array([0.0, 0.0, 0.0])
            result['pointing_vector'] = np.array([0.0, 0.0, 0.0])

        return result

def split_pointing_data(data_dir, output_dir, train_ratio=0.7,
                       val_ratio=0.15, test_ratio=0.15,
                       stratify=True, seed=42, use_test=False):
    """Split your data into train/val/test"""

    # Find samples
    samples = []
    for jpg in Path(data_dir).glob("*.jpg"):
        base = jpg.stem
        npy = Path(data_dir) / f"{base}.npy"
        txt = Path(data_dir) / f"{base}.txt"
        if npy.exists() and txt.exists():
            samples.append({'jpg': jpg, 'npy': npy, 'txt': txt})

    print(f"Found {len(samples)} samples")

    # Get labels
    def get_label(txt):
        with open(txt) as f:
            return int(f.readline().strip())

    # Stratified split
    if stratify:
        pointing = [s for s in samples if get_label(s['txt']) == 1]
        not_pointing = [s for s in samples if get_label(s['txt']) == 0]

        random.seed(seed)
        random.shuffle(pointing)
        random.shuffle(not_pointing)

        def split(lst):
            if use_test:
                n = len(lst)
                t = int(n * train_ratio)
                v = t + int(n * val_ratio)
                return lst[:t], lst[t:v], lst[v:]
            else:
                n = len(lst)
                t = int(n * train_ratio)
                return lst[:t], lst[t:], []

        p_t, p_v, p_te = split(pointing)
        np_t, np_v, np_te = split(not_pointing)

        train = p_t + np_t
        val = p_v + np_v
        test = p_te + np_te

        for s in [train, val, test]:
            random.shuffle(s)
    else:
        random.seed(seed)
        random.shuffle(samples)
        n = len(samples)
        t = int(n * train_ratio)
        v = t + int(n * val_ratio)
        train, val, test = samples[:t], samples[t:v], samples[v:]

    # Copy files
    for name, split in [('train', train), ('val', val), ('test', test)]:
        d = Path(output_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        for s in split:
            for k in ['jpg', 'npy', 'txt']:
                shutil.copy2(s[k], d / s[k].name)
        print(f"✓ {name}: {len(split)} samples")

    return {'train': len(train), 'val': len(val), 'test': len(test)}

# Example usage
if __name__ == "__main__":
    # Create dataset
<<<<<<< HEAD
    split_pointing_data("data", "split_data", train_ratio=0.85,val_ratio=0.15)
=======
    split_pointing_data("data", "split_data", train_ratio=0.85,val_ratio=0.15)
>>>>>>> 85dfa7a61e7f0e9b1e04645a915096120e90aea6
