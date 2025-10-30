import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, Optional


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
    
    def __init__(self, data_dir: str, transform: Optional[callable] = None):
        """
        Args:
            data_dir: Path to directory containing .jpg, .npy, and .txt files
            transform: Optional transform to apply to the image
        """
        self.data_dir = data_dir
        self.transform = transform
        
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
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            rgbd_image: torch.Tensor of shape (4, H, W) - RGB-D concatenated
            label_dict: dict containing {
                'is_pointing': int (0 or 1),
                'wrist_coords': torch.Tensor (3,) or None,
                'pointing_vector': torch.Tensor (3,) or None
            }
        """
        sample = self.samples[idx]
        
        # Load RGB image (BGR -> RGB)
        bgr_img = cv2.imread(sample['image_path'])
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        rgb_img = np.transpose(rgb_img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        
        # Load depth image
        depth_img = np.load(sample['depth_path'])
        depth_img = depth_img.astype(np.float32) / 1000.0  # Convert mm to meters
        depth_img = np.clip(depth_img, 0, 10.0)  # Clip to 0-10m range
        depth_img = np.expand_dims(depth_img, axis=0)  # Add channel dimension (1, H, W)
        
        # Concatenate RGB + D to get 4D image
        rgbd_image = np.concatenate([rgb_img, depth_img], axis=0)  # (4, H, W)
        
        # Apply transform if provided
        if self.transform:
            rgbd_image = self.transform(rgbd_image)
        
        # Load labels
        label_dict = self._load_label(sample['label_path'])
        
        # Convert to torch tensors
        rgbd_image = torch.from_numpy(rgbd_image).float()
        if label_dict['wrist_coords'] is not None:
            label_dict['wrist_coords'] = torch.from_numpy(label_dict['wrist_coords']).float()
        if label_dict['pointing_vector'] is not None:
            label_dict['pointing_vector'] = torch.from_numpy(label_dict['pointing_vector']).float()
        
        return rgbd_image, label_dict
    
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
            result['wrist_coords'] = None
            result['pointing_vector'] = None
        
        return result


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = PointingDataset(data_dir="data")
    
    if len(dataset) > 0:
        # Get first sample
        print(f"First sample files:")
        print(f"  Image: {dataset.samples[0]['image_path']}")
        print(f"  Depth: {dataset.samples[0]['depth_path']}")
        print(f"  Label: {dataset.samples[0]['label_path']}")
        print()
        
        rgbd_image, label = dataset[0]
        
        print(f"RGB-D Image shape: {rgbd_image.shape}")
        print(f"Label: {label}")
        print(f"Is pointing: {label['is_pointing']}")
        
        if label['is_pointing'] == 1:
            print(f"Wrist coords: {label['wrist_coords']}")
            print(f"Pointing vector: {label['pointing_vector']}")
        
        print("\n" + "="*50)
        print("Dataset loaded successfully!")
        print("="*50)
        
        # Test with DataLoader - skip the labels batching for now
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Just get the first batch to verify images work
        images, labels = next(iter(dataloader))
        print(f"\nBatch of {len(labels)} images:")
        print(f"  Images shape: {images.shape}")  # Should be (batch_size, 4, 1080, 1920)
        print(f"  Successfully loaded {len(labels)} samples")
        
    else:
        print("No data found. Run collect_data.py to collect samples.")