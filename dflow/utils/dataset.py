"""
Dataset utilities and custom dataset implementations for dFlow.

This module provides various dataset classes and utilities for loading
and preprocessing data for diffusion model training.
"""

import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Callable, Dict, Any


class CustomDataset(Dataset):
    """
    Custom dataset class for loading images from a directory.
    
    This dataset can load images from a folder structure and apply
    custom transforms for diffusion model training.
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 image_extensions: List[str] = None,
                 class_to_idx: Optional[Dict[str, int]] = None,
                 create_class_mapping: bool = True):
        """
        Initialize custom dataset.
        
        Args:
            data_path: Path to the dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            image_extensions: List of valid image extensions
            class_to_idx: Mapping from class names to indices
            create_class_mapping: Whether to create class mapping from folder structure
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all image files
        self.samples = self._find_samples()
        
        # Create or use provided class mapping
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        elif create_class_mapping:
            self.class_to_idx = self._create_class_mapping()
        else:
            self.class_to_idx = None
        
        # Create class names list
        if self.class_to_idx is not None:
            self.classes = list(self.class_to_idx.keys())
            self.num_classes = len(self.classes)
        else:
            self.classes = None
            self.num_classes = 0
    
    def _find_samples(self) -> List[Tuple[str, int]]:
        """Find all image samples in the dataset directory."""
        samples = []
        
        if self.data_path.is_dir():
            # If directory contains subdirectories (class-based)
            if any(self.data_path.iterdir()):
                for class_dir in sorted(self.data_path.iterdir()):
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        for img_path in class_dir.iterdir():
                            if img_path.suffix.lower() in self.image_extensions:
                                samples.append((str(img_path), class_name))
            else:
                # If directory contains images directly
                for img_path in sorted(self.data_path.iterdir()):
                    if img_path.suffix.lower() in self.image_extensions:
                        samples.append((str(img_path), 0))  # Single class
        
        return samples
    
    def _create_class_mapping(self) -> Dict[str, int]:
        """Create class mapping from folder structure."""
        class_names = set()
        for _, class_name in self.samples:
            class_names.add(class_name)
        
        return {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item from dataset."""
        img_path, class_name = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get class index
        if self.class_to_idx is not None:
            target = self.class_to_idx[class_name]
        else:
            target = 0
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target


class ImageNetDataset(Dataset):
    """
    ImageNet dataset wrapper with custom transforms.
    
    This provides a convenient wrapper around torchvision's ImageNet
    dataset with custom transforms for diffusion model training.
    """
    
    def __init__(self, 
                 root: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        """
        Initialize ImageNet dataset.
        
        Args:
            root: Root directory of the dataset
            split: Dataset split ('train', 'val')
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            download: Whether to download the dataset
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Create torchvision ImageNet dataset
        self.dataset = torchvision.datasets.ImageNet(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.num_classes = len(self.classes)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class CIFARDataset(Dataset):
    """
    CIFAR dataset wrapper with custom transforms.
    
    This provides a convenient wrapper around torchvision's CIFAR
    datasets with custom transforms for diffusion model training.
    """
    
    def __init__(self, 
                 root: str,
                 dataset_name: str = 'CIFAR10',
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True):
        """
        Initialize CIFAR dataset.
        
        Args:
            root: Root directory of the dataset
            dataset_name: Name of the dataset ('CIFAR10' or 'CIFAR100')
            train: Whether to use training set
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            download: Whether to download the dataset
        """
        self.root = root
        self.dataset_name = dataset_name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        # Create torchvision CIFAR dataset
        if dataset_name.upper() == 'CIFAR10':
            self.dataset = torchvision.datasets.CIFAR10(
                root=root,
                train=train,
                transform=transform,
                target_transform=target_transform,
                download=download
            )
        elif dataset_name.upper() == 'CIFAR100':
            self.dataset = torchvision.datasets.CIFAR100(
                root=root,
                train=train,
                transform=transform,
                target_transform=target_transform,
                download=download
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.num_classes = len(self.classes)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class DiffusionDataset(Dataset):
    """
    Dataset specifically designed for diffusion model training.
    
    This dataset handles the specific requirements of diffusion models,
    including timestep sampling and noise generation.
    """
    
    def __init__(self, 
                 base_dataset: Dataset,
                 noise_schedule: str = 'linear',
                 num_timesteps: int = 1000,
                 image_size: int = 256,
                 normalize: bool = True):
        """
        Initialize diffusion dataset.
        
        Args:
            base_dataset: Base dataset to wrap
            noise_schedule: Noise schedule type ('linear', 'cosine', 'sigmoid')
            num_timesteps: Number of diffusion timesteps
            image_size: Target image size
            normalize: Whether to normalize images to [-1, 1]
        """
        self.base_dataset = base_dataset
        self.noise_schedule = noise_schedule
        self.num_timesteps = num_timesteps
        self.image_size = image_size
        self.normalize = normalize
        
        # Create noise schedule
        self.betas = self._create_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _create_noise_schedule(self) -> torch.Tensor:
        """Create noise schedule for diffusion process."""
        if self.noise_schedule == 'linear':
            return torch.linspace(0.0001, 0.02, self.num_timesteps)
        elif self.noise_schedule == 'cosine':
            # Cosine schedule
            s = 0.008
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif self.noise_schedule == 'sigmoid':
            # Sigmoid schedule
            v_start = 0.0001
            v_end = 0.02
            steps = self.num_timesteps + 1
            t = torch.linspace(0, 1, steps)
            v = v_start + (v_end - v_start) * torch.sigmoid(12 * (t - 0.5))
            return v[1:] - v[:-1]
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get item from dataset.
        
        Returns:
            Tuple of (noisy_image, clean_image, timestep, label)
        """
        # Get base sample
        image, label = self.base_dataset[idx]
        
        # Normalize to [-1, 1] if needed
        if self.normalize:
            image = image * 2.0 - 1.0
        
        # Sample random timestep
        t = torch.randint(0, self.num_timesteps, (1,)).item()
        
        # Add noise
        noise = torch.randn_like(image)
        alpha_cumprod_t = self.alphas_cumprod[t]
        noisy_image = torch.sqrt(alpha_cumprod_t) * image + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        return noisy_image, image, torch.tensor(t), label


class PairedDataset(Dataset):
    """
    Dataset for paired data (e.g., for contrastive learning).
    
    This dataset can create pairs of samples for contrastive learning
    or other tasks that require paired data.
    """
    
    def __init__(self, 
                 base_dataset: Dataset,
                 pair_strategy: str = 'random',
                 transform1: Optional[Callable] = None,
                 transform2: Optional[Callable] = None,
                 same_class_prob: float = 0.5):
        """
        Initialize paired dataset.
        
        Args:
            base_dataset: Base dataset to wrap
            pair_strategy: Strategy for creating pairs ('random', 'same_class', 'different_class')
            transform1: Transform for first view
            transform2: Transform for second view
            same_class_prob: Probability of sampling same class for random strategy
        """
        self.base_dataset = base_dataset
        self.pair_strategy = pair_strategy
        self.transform1 = transform1
        self.transform2 = transform2
        self.same_class_prob = same_class_prob
        
        # Create class indices mapping
        self.class_indices = {}
        for idx, (_, label) in enumerate(base_dataset):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get paired item from dataset.
        
        Returns:
            Tuple of (view1, view2, label)
        """
        # Get base sample
        image, label = self.base_dataset[idx]
        
        # Create second view
        if self.pair_strategy == 'random':
            if torch.rand(1) < self.same_class_prob:
                # Same class
                if len(self.class_indices[label]) > 1:
                    idx2 = torch.randint(0, len(self.class_indices[label]), (1,)).item()
                    idx2 = self.class_indices[label][idx2]
                    if idx2 == idx:
                        idx2 = self.class_indices[label][(idx2 + 1) % len(self.class_indices[label])]
                else:
                    idx2 = idx
            else:
                # Different class
                other_classes = [c for c in self.class_indices.keys() if c != label]
                if other_classes:
                    other_class = torch.randint(0, len(other_classes), (1,)).item()
                    other_class = other_classes[other_class]
                    idx2 = torch.randint(0, len(self.class_indices[other_class]), (1,)).item()
                    idx2 = self.class_indices[other_class][idx2]
                else:
                    idx2 = idx
        elif self.pair_strategy == 'same_class':
            if len(self.class_indices[label]) > 1:
                idx2 = torch.randint(0, len(self.class_indices[label]), (1,)).item()
                idx2 = self.class_indices[label][idx2]
                if idx2 == idx:
                    idx2 = self.class_indices[label][(idx2 + 1) % len(self.class_indices[label])]
            else:
                idx2 = idx
        elif self.pair_strategy == 'different_class':
            other_classes = [c for c in self.class_indices.keys() if c != label]
            if other_classes:
                other_class = torch.randint(0, len(other_classes), (1,)).item()
                other_class = other_classes[other_class]
                idx2 = torch.randint(0, len(self.class_indices[other_class]), (1,)).item()
                idx2 = self.class_indices[other_class][idx2]
            else:
                idx2 = idx
        else:
            raise ValueError(f"Unknown pair strategy: {self.pair_strategy}")
        
        # Get second sample
        image2, label2 = self.base_dataset[idx2]
        
        # Apply transforms
        if self.transform1 is not None:
            image = self.transform1(image)
        if self.transform2 is not None:
            image2 = self.transform2(image2)
        
        return image, image2, label


def create_dataloader(dataset: Dataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     drop_last: bool = False,
                     **kwargs) -> DataLoader:
    """
    Create a DataLoader with common settings.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )


def get_dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        'length': len(dataset),
        'num_classes': getattr(dataset, 'num_classes', None),
        'classes': getattr(dataset, 'classes', None),
        'class_to_idx': getattr(dataset, 'class_to_idx', None),
    }
    
    # Get sample info
    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            info['sample_shape'] = sample[0].shape if hasattr(sample[0], 'shape') else None
            info['target_type'] = type(sample[1]).__name__
    
    return info


def split_dataset(dataset: Dataset, 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
