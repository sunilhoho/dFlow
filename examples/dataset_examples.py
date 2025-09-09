"""
Examples demonstrating dFlow dataset utilities.

This script shows how to use the various dataset classes and utilities
provided by the dFlow library.
"""

import torch
import torchvision
from dflow.utils import (
    CustomDataset, ImageNetDataset, CIFARDataset, DiffusionDataset, PairedDataset,
    get_transforms, create_augmentation_pipeline, create_dataloader,
    get_dataset_info, analyze_dataset_distribution, visualize_dataset_samples
)


def example_custom_dataset():
    """Example using CustomDataset for loading images from a directory."""
    print("=== Custom Dataset Example ===")
    
    # Create transforms
    transform = get_transforms('custom', 'train', image_size=256, augmentation=True)
    
    # Create custom dataset
    dataset = CustomDataset(
        data_path='/path/to/your/images',  # Update this path
        split='train',
        transform=transform,
        image_extensions=['.jpg', '.jpeg', '.png', '.bmp']
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Classes: {dataset.classes}")
    
    # Get dataset info
    info = get_dataset_info(dataset)
    print(f"Dataset info: {info}")
    
    # Analyze distribution
    stats = analyze_dataset_distribution(dataset)
    print(f"Distribution stats: {stats}")
    
    print()


def example_cifar_dataset():
    """Example using CIFAR dataset."""
    print("=== CIFAR Dataset Example ===")
    
    # Create transforms
    transform = get_transforms('cifar10', 'train', image_size=32, augmentation=True)
    
    # Create CIFAR-10 dataset
    dataset = CIFARDataset(
        root='/tmp/cifar10',  # Will download if not exists
        dataset_name='CIFAR10',
        train=True,
        transform=transform,
        download=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Classes: {dataset.classes[:5]}...")  # Show first 5 classes
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    print(f"Dataloader batches: {len(dataloader)}")
    
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    images, labels = sample_batch
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    print()


def example_diffusion_dataset():
    """Example using DiffusionDataset for diffusion model training."""
    print("=== Diffusion Dataset Example ===")
    
    # Create base dataset
    base_dataset = CIFARDataset(
        root='/tmp/cifar10',
        dataset_name='CIFAR10',
        train=True,
        transform=get_transforms('cifar10', 'train', image_size=32, augmentation=False),
        download=True
    )
    
    # Wrap with DiffusionDataset
    diffusion_dataset = DiffusionDataset(
        base_dataset=base_dataset,
        noise_schedule='linear',
        num_timesteps=1000,
        image_size=32,
        normalize=True
    )
    
    print(f"Diffusion dataset length: {len(diffusion_dataset)}")
    
    # Get a sample
    sample = diffusion_dataset[0]
    noisy_image, clean_image, timestep, label = sample
    
    print(f"Noisy image shape: {noisy_image.shape}")
    print(f"Clean image shape: {clean_image.shape}")
    print(f"Timestep: {timestep}")
    print(f"Label: {label}")
    print(f"Image range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
    
    print()


def example_paired_dataset():
    """Example using PairedDataset for contrastive learning."""
    print("=== Paired Dataset Example ===")
    
    # Create base dataset
    base_dataset = CIFARDataset(
        root='/tmp/cifar10',
        dataset_name='CIFAR10',
        train=True,
        transform=get_transforms('cifar10', 'train', image_size=32, augmentation=False),
        download=True
    )
    
    # Create paired dataset
    paired_dataset = PairedDataset(
        base_dataset=base_dataset,
        pair_strategy='random',
        transform1=get_transforms('cifar10', 'train', image_size=32, augmentation=True),
        transform2=get_transforms('cifar10', 'train', image_size=32, augmentation=True),
        same_class_prob=0.5
    )
    
    print(f"Paired dataset length: {len(paired_dataset)}")
    
    # Get a sample
    sample = paired_dataset[0]
    view1, view2, label = sample
    
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    print(f"Label: {label}")
    
    print()


def example_custom_transforms():
    """Example using custom transform pipelines."""
    print("=== Custom Transforms Example ===")
    
    # Create custom augmentation pipeline
    strong_augmentation = create_augmentation_pipeline(
        image_size=256,
        strength='strong',
        include_geometric=True,
        include_color=True,
        include_cutout=True
    )
    
    print("Strong augmentation pipeline:")
    for i, transform in enumerate(strong_augmentation.transforms):
        print(f"  {i+1}. {transform}")
    
    # Create light augmentation pipeline
    light_augmentation = create_augmentation_pipeline(
        image_size=128,
        strength='light',
        include_geometric=True,
        include_color=False,
        include_cutout=False
    )
    
    print("\nLight augmentation pipeline:")
    for i, transform in enumerate(light_augmentation.transforms):
        print(f"  {i+1}. {transform}")
    
    print()


def example_dataset_analysis():
    """Example analyzing dataset properties."""
    print("=== Dataset Analysis Example ===")
    
    # Create a dataset
    dataset = CIFARDataset(
        root='/tmp/cifar10',
        dataset_name='CIFAR10',
        train=True,
        transform=get_transforms('cifar10', 'train', image_size=32, augmentation=False),
        download=True
    )
    
    # Analyze distribution
    stats = analyze_dataset_distribution(dataset)
    print("Distribution Statistics:")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Is balanced: {stats['is_balanced']}")
    print(f"  Min class count: {stats['min_class_count']}")
    print(f"  Max class count: {stats['max_class_count']}")
    print(f"  Mean class count: {stats['mean_class_count']:.2f}")
    print(f"  Std class count: {stats['std_class_count']:.2f}")
    
    # Show class distribution
    print("\nClass Distribution:")
    for class_idx, count in stats['class_counts'].items():
        percentage = stats['class_distribution'][class_idx] * 100
        print(f"  Class {class_idx}: {count} samples ({percentage:.1f}%)")
    
    print()


def example_dataloader_creation():
    """Example creating dataloaders with different configurations."""
    print("=== DataLoader Creation Example ===")
    
    # Create dataset
    dataset = CIFARDataset(
        root='/tmp/cifar10',
        dataset_name='CIFAR10',
        train=True,
        transform=get_transforms('cifar10', 'train', image_size=32, augmentation=True),
        download=True
    )
    
    # Create different dataloaders
    configs = [
        {'batch_size': 32, 'shuffle': True, 'num_workers': 4},
        {'batch_size': 64, 'shuffle': False, 'num_workers': 2},
        {'batch_size': 16, 'shuffle': True, 'num_workers': 0},
    ]
    
    for i, config in enumerate(configs):
        dataloader = create_dataloader(dataset, **config)
        print(f"DataLoader {i+1}: {config}")
        print(f"  Batches: {len(dataloader)}")
        print(f"  Total samples: {len(dataloader.dataset)}")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Workers: {dataloader.num_workers}")
        print()
    
    print()


def example_visualization():
    """Example visualizing dataset samples."""
    print("=== Dataset Visualization Example ===")
    
    # Create dataset
    dataset = CIFARDataset(
        root='/tmp/cifar10',
        dataset_name='CIFAR10',
        train=True,
        transform=get_transforms('cifar10', 'train', image_size=32, augmentation=False),
        download=True
    )
    
    print("Visualizing dataset samples...")
    print("(This would show a plot in a Jupyter notebook or GUI environment)")
    
    # In a real environment, this would show the plot
    # visualize_dataset_samples(dataset, num_samples=8, figsize=(12, 8))
    
    print("Visualization complete!")
    print()


if __name__ == "__main__":
    print("dFlow Dataset Utilities Examples")
    print("=" * 50)
    print()
    
    example_custom_dataset()
    example_cifar_dataset()
    example_diffusion_dataset()
    example_paired_dataset()
    example_custom_transforms()
    example_dataset_analysis()
    example_dataloader_creation()
    example_visualization()
    
    print("All examples completed!")
    print("\nNote: Some examples require actual data paths and may download datasets.")
    print("Update the paths in the examples to use with your own data.")
