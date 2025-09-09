"""
Transform utilities for dFlow datasets.

This module provides various transform functions and utilities
for data preprocessing and augmentation.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional, Union, Callable


def get_transforms(dataset_name: str = 'imagenet',
                  split: str = 'train',
                  image_size: int = 256,
                  normalize: bool = True,
                  augmentation: bool = True) -> transforms.Compose:
    """
    Get standard transforms for different datasets.
    
    Args:
        dataset_name: Name of the dataset ('imagenet', 'cifar10', 'cifar100', 'custom')
        split: Dataset split ('train', 'val', 'test')
        image_size: Target image size
        normalize: Whether to normalize images
        augmentation: Whether to apply data augmentation
        
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if dataset_name.lower() == 'imagenet':
        if split == 'train' and augmentation:
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        
        if normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
    
    elif dataset_name.lower() in ['cifar10', 'cifar100']:
        if split == 'train' and augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(image_size, padding=4),
                transforms.ToTensor(),
            ])
        else:
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        
        if normalize:
            if dataset_name.lower() == 'cifar10':
                transform_list.append(transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ))
            else:  # cifar100
                transform_list.append(transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                ))
    
    elif dataset_name.lower() == 'custom':
        if split == 'train' and augmentation:
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        
        if normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ))
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transforms.Compose(transform_list)


def create_augmentation_pipeline(image_size: int = 256,
                                strength: str = 'medium',
                                include_geometric: bool = True,
                                include_color: bool = True,
                                include_cutout: bool = False) -> transforms.Compose:
    """
    Create a custom augmentation pipeline.
    
    Args:
        image_size: Target image size
        strength: Augmentation strength ('light', 'medium', 'strong')
        include_geometric: Whether to include geometric augmentations
        include_color: Whether to include color augmentations
        include_cutout: Whether to include cutout augmentation
        
    Returns:
        Composed transforms
    """
    transform_list = [transforms.Resize((image_size, image_size))]
    
    if strength == 'light':
        if include_geometric:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
        if include_color:
            transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
    
    elif strength == 'medium':
        if include_geometric:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        if include_color:
            transform_list.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ))
    
    elif strength == 'strong':
        if include_geometric:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        if include_color:
            transform_list.extend([
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
                ),
                transforms.RandomGrayscale(p=0.1),
            ])
    
    if include_cutout:
        transform_list.append(Cutout(n_holes=1, length=image_size//4))
    
    transform_list.append(transforms.ToTensor())
    
    return transforms.Compose(transform_list)


class Cutout:
    """
    Cutout augmentation.
    
    Randomly mask out rectangular regions of the image.
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        """
        Initialize cutout augmentation.
        
        Args:
            n_holes: Number of holes to cut out
            length: Length of each hole
        """
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply cutout augmentation.
        
        Args:
            img: Input image
            
        Returns:
            Augmented image
        """
        h, w = img.size[1], img.size[0]  # PIL uses (width, height)
        mask = np.ones((h, w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img


class RandomErasing:
    """
    Random erasing augmentation.
    
    Randomly erase rectangular regions of the image.
    """
    
    def __init__(self, 
                 p: float = 0.5,
                 scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3),
                 value: Union[float, str] = 0):
        """
        Initialize random erasing.
        
        Args:
            p: Probability of applying random erasing
            scale: Range of area scale
            ratio: Range of aspect ratio
            value: Value to fill erased area
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random erasing.
        
        Args:
            img: Input tensor
            
        Returns:
            Augmented tensor
        """
        if torch.rand(1) > self.p:
            return img
        
        c, h, w = img.shape
        area = h * w
        
        for _ in range(10):  # Try up to 10 times
            target_area = torch.empty(1).uniform_(self.scale[0], self.scale[1]) * area
            aspect_ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1])
            
            erasing_h = int(round(torch.sqrt(target_area * aspect_ratio).item()))
            erasing_w = int(round(torch.sqrt(target_area / aspect_ratio).item()))
            
            if erasing_h < h and erasing_w < w:
                y = torch.randint(0, h - erasing_h + 1, size=(1,)).item()
                x = torch.randint(0, w - erasing_w + 1, size=(1,)).item()
                
                if self.value == 'random':
                    img[:, y:y + erasing_h, x:x + erasing_w] = torch.randn(c, erasing_h, erasing_w)
                else:
                    img[:, y:y + erasing_h, x:x + erasing_w] = self.value
                
                break
        
        return img


class MixUp:
    """
    MixUp augmentation.
    
    Mix two images and their labels.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch.
        
        Args:
            batch: Tuple of (images, labels)
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lam)
        """
        images, labels = batch
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index, :]
        
        return mixed_images, labels, labels[index], lam


class CutMix:
    """
    CutMix augmentation.
    
    Cut and paste regions between images.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to a batch.
        
        Args:
            batch: Tuple of (images, labels)
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lam)
        """
        images, labels = batch
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Get bounding box
        W = images.size(3)
        H = images.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Mix images
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return mixed_images, labels, labels[index], lam
