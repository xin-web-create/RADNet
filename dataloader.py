"""
Data loading utilities for RADNet
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class DehazingDataset(Dataset):
    """Dataset for image dehazing"""
    
    def __init__(self, hazy_dir, clear_dir=None, transform=None, img_size=(256, 256)):
        """
        Args:
            hazy_dir: Directory containing hazy images
            clear_dir: Directory containing clear (ground truth) images
            transform: Optional transform to be applied on images
            img_size: Resize images to this size
        """
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.img_size = img_size
        
        if not os.path.exists(hazy_dir):
            raise ValueError(f"Hazy image directory does not exist: {hazy_dir}")
        
        self.hazy_images = sorted([f for f in os.listdir(hazy_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        if clear_dir is not None:
            if not os.path.exists(clear_dir):
                raise ValueError(f"Clear image directory does not exist: {clear_dir}")
            self.clear_images = sorted([f for f in os.listdir(clear_dir) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        else:
            self.clear_images = None
            
    def __len__(self):
        return len(self.hazy_images)
    
    def __getitem__(self, idx):
        # Load hazy image
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        hazy_img = Image.open(hazy_path).convert('RGB')
        
        if self.transform:
            hazy_img = self.transform(hazy_img)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])
            hazy_img = transform(hazy_img)
        
        # Load clear image if available
        if self.clear_images is not None:
            clear_path = os.path.join(self.clear_dir, self.clear_images[idx])
            clear_img = Image.open(clear_path).convert('RGB')
            
            if self.transform:
                clear_img = self.transform(clear_img)
            else:
                transform = transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                ])
                clear_img = transform(clear_img)
            
            return hazy_img, clear_img
        else:
            return hazy_img


def get_train_transform(img_size=(256, 256)):
    """Get training data transforms with augmentation"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])


def get_val_transform(img_size=(256, 256)):
    """Get validation/test data transforms"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])


def create_dataloaders(train_hazy_dir, train_clear_dir, 
                       val_hazy_dir=None, val_clear_dir=None,
                       batch_size=16, num_workers=4, img_size=(256, 256)):
    """
    Create training and validation dataloaders
    
    Args:
        train_hazy_dir: Directory with training hazy images
        train_clear_dir: Directory with training clear images
        val_hazy_dir: Directory with validation hazy images
        val_clear_dir: Directory with validation clear images
        batch_size: Batch size
        num_workers: Number of worker threads
        img_size: Image size for resizing
        
    Returns:
        train_loader, val_loader (or just train_loader if no validation data)
    """
    # Training dataset and loader
    train_dataset = DehazingDataset(
        train_hazy_dir, 
        train_clear_dir,
        transform=get_train_transform(img_size),
        img_size=img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation dataset and loader (if provided)
    if val_hazy_dir is not None and val_clear_dir is not None:
        val_dataset = DehazingDataset(
            val_hazy_dir,
            val_clear_dir,
            transform=get_val_transform(img_size),
            img_size=img_size
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    else:
        return train_loader, None
