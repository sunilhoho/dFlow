"""
Data utilities for dFlow.

This module provides various utility functions for data loading,
preprocessing, and analysis.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image


def create_dataloader_from_config(config: Dict[str, Any]) -> DataLoader:
    """
    Create a DataLoader from configuration dictionary.
    
    Args:
        config: Configuration dictionary containing dataset and dataloader parameters
        
    Returns:
        DataLoader instance
    """
    from .dataset import CustomDataset, ImageNetDataset, CIFARDataset, DiffusionDataset
    from .transforms import get_transforms
    
    # Get dataset parameters
    dataset_name = config.get('_target_', 'CustomDataset')
    data_path = config.get('root', config.get('data_path', '/path/to/data'))
    split = config.get('split', 'train')
    image_size = config.get('image_size', 256)
    normalize = config.get('normalize', True)
    augmentation = config.get('augmentation', True)
    
    # Create transforms
    transform = get_transforms(
        dataset_name=dataset_name.lower(),
        split=split,
        image_size=image_size,
        normalize=normalize,
        augmentation=augmentation
    )
    
    # Create dataset
    if dataset_name == 'CustomDataset':
        dataset = CustomDataset(
            data_path=data_path,
            split=split,
            transform=transform,
            image_extensions=config.get('image_extensions', ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
        )
    elif dataset_name == 'ImageNetDataset':
        dataset = ImageNetDataset(
            root=data_path,
            split=split,
            transform=transform,
            download=config.get('download', False)
        )
    elif dataset_name in ['CIFARDataset', 'CIFAR10', 'CIFAR100']:
        dataset = CIFARDataset(
            root=data_path,
            dataset_name=dataset_name,
            train=(split == 'train'),
            transform=transform,
            download=config.get('download', True)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Wrap with DiffusionDataset if needed
    if config.get('diffusion_dataset', False):
        dataset = DiffusionDataset(
            base_dataset=dataset,
            noise_schedule=config.get('noise_schedule', 'linear'),
            num_timesteps=config.get('num_timesteps', 1000),
            image_size=image_size,
            normalize=normalize
        )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=config.get('shuffle', True),
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        drop_last=config.get('drop_last', True)
    )
    
    return dataloader


def visualize_dataset_samples(dataset: Dataset,
                             num_samples: int = 8,
                             figsize: Tuple[int, int] = (12, 8),
                             save_path: Optional[str] = None) -> None:
    """
    Visualize samples from a dataset.
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to show
        figsize: Figure size
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            image, label = sample[0], sample[1]
        else:
            image, label = sample, 0
        
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.size(0) == 3:  # CHW format
                image = image.permute(1, 2, 0)
            image = image.numpy()
            
            # Denormalize if needed
            if image.min() < 0:  # Assuming normalized to [-1, 1]
                image = (image + 1) / 2
            image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def analyze_dataset_distribution(dataset: Dataset) -> Dict[str, Any]:
    """
    Analyze the class distribution of a dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary containing distribution statistics
    """
    class_counts = {}
    total_samples = len(dataset)
    
    # Count samples per class
    for i in range(total_samples):
        sample = dataset[i]
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            label = sample[1]
        else:
            label = 0
        
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Calculate statistics
    class_names = list(class_counts.keys())
    class_counts_list = list(class_counts.values())
    
    stats = {
        'num_classes': len(class_names),
        'total_samples': total_samples,
        'class_counts': class_counts,
        'class_distribution': {k: v/total_samples for k, v in class_counts.items()},
        'min_class_count': min(class_counts_list),
        'max_class_count': max(class_counts_list),
        'mean_class_count': np.mean(class_counts_list),
        'std_class_count': np.std(class_counts_list),
        'is_balanced': max(class_counts_list) - min(class_counts_list) < 0.1 * total_samples
    }
    
    return stats


def create_class_weights(dataset: Dataset, 
                        method: str = 'balanced') -> torch.Tensor:
    """
    Create class weights for handling imbalanced datasets.
    
    Args:
        dataset: Dataset to analyze
        method: Method for creating weights ('balanced', 'inverse', 'sqrt')
        
    Returns:
        Tensor of class weights
    """
    stats = analyze_dataset_distribution(dataset)
    class_counts = stats['class_counts']
    num_classes = stats['num_classes']
    
    # Create weight tensor
    weights = torch.zeros(num_classes)
    
    if method == 'balanced':
        # Balanced weights (inverse frequency)
        total_samples = stats['total_samples']
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (num_classes * count)
    
    elif method == 'inverse':
        # Inverse frequency
        max_count = max(class_counts.values())
        for class_idx, count in class_counts.items():
            weights[class_idx] = max_count / count
    
    elif method == 'sqrt':
        # Square root of inverse frequency
        max_count = max(class_counts.values())
        for class_idx, count in class_counts.items():
            weights[class_idx] = np.sqrt(max_count / count)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights


def save_dataset_info(dataset: Dataset, 
                     save_path: str,
                     include_samples: bool = True,
                     num_sample_images: int = 16) -> None:
    """
    Save dataset information to a JSON file.
    
    Args:
        dataset: Dataset to analyze
        save_path: Path to save the information
        include_samples: Whether to include sample images
        num_sample_images: Number of sample images to save
    """
    info = {
        'dataset_info': get_dataset_info(dataset),
        'distribution_stats': analyze_dataset_distribution(dataset),
        'class_weights': create_class_weights(dataset).tolist()
    }
    
    if include_samples:
        # Save sample images
        sample_dir = Path(save_path).parent / 'sample_images'
        sample_dir.mkdir(exist_ok=True)
        
        for i in range(min(num_sample_images, len(dataset))):
            sample = dataset[i]
            if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                image, label = sample[0], sample[1]
            else:
                image, label = sample, 0
            
            # Convert tensor to PIL Image
            if isinstance(image, torch.Tensor):
                if image.dim() == 3 and image.size(0) == 3:  # CHW format
                    image = image.permute(1, 2, 0)
                image = image.numpy()
                
                # Denormalize if needed
                if image.min() < 0:  # Assuming normalized to [-1, 1]
                    image = (image + 1) / 2
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            image.save(sample_dir / f'sample_{i:04d}_label_{label}.png')
        
        info['sample_images_dir'] = str(sample_dir)
    
    # Save info to JSON
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)


def load_dataset_info(load_path: str) -> Dict[str, Any]:
    """
    Load dataset information from a JSON file.
    
    Args:
        load_path: Path to load the information from
        
    Returns:
        Dictionary containing dataset information
    """
    with open(load_path, 'r') as f:
        info = json.load(f)
    
    return info


def create_balanced_sampler(dataset: Dataset) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a balanced sampler for imbalanced datasets.
    
    Args:
        dataset: Dataset to create sampler for
        
    Returns:
        WeightedRandomSampler instance
    """
    class_weights = create_class_weights(dataset)
    
    # Get sample weights
    sample_weights = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            label = sample[1]
        else:
            label = 0
        
        sample_weights.append(class_weights[label].item())
    
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )


def get_dataloader_stats(dataloader: DataLoader) -> Dict[str, Any]:
    """
    Get statistics about a DataLoader.
    
    Args:
        dataloader: DataLoader to analyze
        
    Returns:
        Dictionary containing DataLoader statistics
    """
    stats = {
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'total_samples': len(dataloader.dataset),
        'num_workers': dataloader.num_workers,
        'pin_memory': dataloader.pin_memory,
        'drop_last': dataloader.drop_last,
        'shuffle': dataloader.sampler is None and dataloader.batch_sampler is None
    }
    
    # Get sample batch info
    if len(dataloader) > 0:
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
            images, labels = sample_batch[0], sample_batch[1]
            stats['sample_batch_shape'] = images.shape
            stats['sample_label_shape'] = labels.shape if hasattr(labels, 'shape') else None
    
    return stats
