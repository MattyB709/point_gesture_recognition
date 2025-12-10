"""
Modified PointingDataset that works with torchvision transforms.

Key changes:
1. Keeps image as PIL until after transform
2. Only converts to tensor after transform (or in transform itself)
3. Compatible with standard torchvision transforms
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from pathlib import Path
import random
import math
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class ViTDataset(Dataset):
    """
    Dataset that works with torchvision transforms.
    Keeps images as PIL until after transform is applied.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        transform: Optional[callable] = None,
        augment: bool = True,
        # Augmentation parameters (for direction vector)
        horizontal_flip_prob: float = 0.5,
    ):
        """
        Args:
            data_dir: Path to directory containing .jpg and .txt files
            transform: Torchvision transforms (should include Resize, ToTensor, Normalize)
            augment: Whether to apply augmentation to direction vectors
            horizontal_flip_prob: Probability of horizontal flip (applies to both image and direction)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.horizontal_flip_prob = horizontal_flip_prob
        
        # Find all samples
        self.samples = []
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg'):
                    base_name = filename[:-4]
                    sample = {
                        'base_name': base_name,
                        'image_path': os.path.join(data_dir, filename),
                        'label_path': os.path.join(data_dir, base_name + '.txt')
                    }
                    self.samples.append(sample)
        
        aug_status = "ON" if augment else "OFF"
        print(f"PointingDatasetWithTransforms: {len(self.samples)} samples, augmentation {aug_status}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        
        # Load image as PIL Image
        pil_img = Image.open(sample['image_path']).convert('RGB')
        
        # Load labels
        label_dict = self._load_label(sample['label_path'])
        is_pointing = label_dict['is_pointing'] == 1
        direction = torch.from_numpy(label_dict['pointing_vector']).float()
        
        # Apply augmentation (horizontal flip to both image and direction)
        if self.augment and random.random() < self.horizontal_flip_prob:
            pil_img = TF.hflip(pil_img)
            if is_pointing:
                direction[0] = -direction[0]  # Flip X component
        
        # Apply transform (Resize, ToTensor, Normalize)
        if self.transform:
            image = self.transform(pil_img)
        else:
            # Default: just convert to tensor
            image = TF.to_tensor(pil_img)
        
        label_dict['pointing_vector'] = direction
        
        return image, label_dict
    
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

class ViTDatasetAggressive(Dataset):
    """
    Dataset with aggressive augmentation for ViT to prevent overfitting.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        transform: Optional[callable] = None,
        augment: bool = True,
        # Augmentation probabilities
        horizontal_flip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
        rotation_prob: float = 0.3,
        perspective_prob: float = 0.3,
        blur_prob: float = 0.3,
        erasing_prob: float = 0.3,
        # Augmentation parameters
        rotation_degrees: float = 15.0,
        color_jitter_strength: float = 0.4,
    ):
        """
        Args:
            augment: Enable augmentation (train=True, val=False)
            horizontal_flip_prob: Probability of horizontal flip
            color_jitter_prob: Probability of color jitter
            rotation_prob: Probability of rotation (NOT recommended for pointing)
            perspective_prob: Probability of perspective transform
            blur_prob: Probability of Gaussian blur
            erasing_prob: Probability of random erasing
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        
        # Augmentation probabilities
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.rotation_prob = rotation_prob
        self.perspective_prob = perspective_prob
        self.blur_prob = blur_prob
        self.erasing_prob = erasing_prob
        
        # Augmentation parameters
        self.rotation_degrees = rotation_degrees
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=color_jitter_strength * 0.5
        )
        
        # Find all samples
        self.samples = []
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg'):
                    base_name = filename[:-4]
                    sample = {
                        'base_name': base_name,
                        'image_path': os.path.join(data_dir, filename),
                        'label_path': os.path.join(data_dir, base_name + '.txt')
                    }
                    self.samples.append(sample)
        
        aug_status = "AGGRESSIVE" if augment else "OFF"
        print(f"ViTDatasetAggressive: {len(self.samples)} samples, augmentation {aug_status}")
        if augment:
            print(f"  Augmentation settings:")
            print(f"    - Horizontal flip: {horizontal_flip_prob:.0%}")
            print(f"    - Color jitter: {color_jitter_prob:.0%}")
            print(f"    - Rotation: {rotation_prob:.0%} (±{rotation_degrees}°)")
            print(f"    - Perspective: {perspective_prob:.0%}")
            print(f"    - Blur: {blur_prob:.0%}")
            print(f"    - Erasing: {erasing_prob:.0%}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        
        # Load image as PIL Image
        pil_img = Image.open(sample['image_path']).convert('RGB')
        
        # Load labels
        label_dict = self._load_label(sample['label_path'])
        is_pointing = label_dict['is_pointing'] == 1
        direction = torch.from_numpy(label_dict['pointing_vector']).float()
        
        # Apply augmentations
        if self.augment:
            pil_img, direction = self._apply_augmentations(pil_img, direction, is_pointing)
        
        # Apply transform (Resize, ToTensor, Normalize)
        if self.transform:
            image = self.transform(pil_img)
        else:
            image = TF.to_tensor(pil_img)
        
        # Random erasing (applied after ToTensor)
        if self.augment and random.random() < self.erasing_prob:
            image = transforms.RandomErasing(p=1.0, scale=(0.02, 0.15))(image)
        
        label_dict['pointing_vector'] = direction
        
        return image, label_dict
    
    def _apply_augmentations(self, img: Image.Image, direction: torch.Tensor, is_pointing: bool):
        """Apply all augmentations to image and direction vector"""
        
        # 1. Horizontal flip (affects direction)
        if random.random() < self.horizontal_flip_prob:
            img = TF.hflip(img)
            if is_pointing:
                direction[0] = -direction[0]
        
        # 2. Color jitter (doesn't affect direction)
        if random.random() < self.color_jitter_prob:
            img = self.color_jitter(img)
        
        # 3. Gaussian blur (doesn't affect direction)
        if random.random() < self.blur_prob:
            kernel_size = random.choice([3, 5, 7])
            img = img.filter(ImageFilter.GaussianBlur(radius=kernel_size/2))
        
        # 4. Random perspective (doesn't significantly affect direction in 3D)
        # if random.random() < self.perspective_prob:
        #     img = TF.perspective(
        #         img,
        #         startpoints=self._get_perspective_params(img.size),
        #         endpoints=[(0, 0), (img.width, 0), (img.width, img.height), (0, img.height)],
        #         interpolation=transforms.InterpolationMode.BILINEAR
            # )
        
        # 5. Small rotation (NOT RECOMMENDED for pointing - commented out)
        # Rotation would require complex 3D rotation matrix for direction
        # if random.random() < self.rotation_prob:
        #     angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        #     img = TF.rotate(img, angle)
        #     # Direction would need rotation matrix transformation
        
        return img, direction
    
    def _get_perspective_params(self, size):
        """Get random perspective distortion parameters"""
        width, height = size
        distortion = 0.1  # 10% distortion
        
        # Randomly distort each corner
        startpoints = [
            (random.uniform(0, width * distortion), random.uniform(0, height * distortion)),
            (width - random.uniform(0, width * distortion), random.uniform(0, height * distortion)),
            (width - random.uniform(0, width * distortion), height - random.uniform(0, height * distortion)),
            (random.uniform(0, width * distortion), height - random.uniform(0, height * distortion))
        ]
        
        return startpoints
    
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
# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    # Define transforms for ViT (matches pretraining)
    vit_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT needs 224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Create datasets with transforms
    train_dataset = ViTDataset(
        "./split_data/train",
        transform=vit_transforms,
        augment=True  # Augments direction vectors
    )
    
    val_dataset = ViTDataset(
        "./split_data/val",
        transform=vit_transforms,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Test loading
    image, labels = train_dataset[0]
    print(f"Image shape: {image.shape}")  # Should be (3, 224, 224)
    print(f"Is pointing: {labels['is_pointing']}")
    print(f"Direction: {labels['pointing_vector']}")