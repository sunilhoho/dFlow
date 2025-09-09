"""
dFlow: Dispersive Flow - A PyTorch library for diffusion models with dispersive loss

This library provides implementations of scalable interpolant transformers (SiT) 
with dispersive loss regularization for improved diffusion-based generative models.

Key Features:
- SiT model implementations with various sizes (S, B, L, XL)
- Dispersive loss regularization for better representation learning
- VICReg loss for self-supervised learning
- Sampling utilities
- Transport-based diffusion methods
- Configurable model architectures
- Comprehensive dataset utilities

Usage:
    import dflow
    from dflow.models import SiT
    from dflow.losses import DispersiveLoss, VICRegLoss
    from dflow.utils import CustomDataset, get_transforms
    from dflow.configs import get_model_config
    
    # Create a model
    config = get_model_config('SiT-XL/2')
    model = SiT(**config)
    
    # Create loss functions
    dispersive_loss = DispersiveLoss(lambda_disp=0.25, tau=1.0)
    vicreg_loss = VICRegLoss(lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0)
    
    # Create dataset
    transform = get_transforms('imagenet', 'train', image_size=256)
    dataset = CustomDataset('/path/to/data', transform=transform)
"""

__version__ = "0.1.0"
__author__ = "dFlow Team"

# Import main components
from .models import *
from .transport import *
from .sampling import *
from .configs import *
from .losses import *
from .utils import *

__all__ = [
    # Models
    'SiT',
    'SiTBlock', 
    'TimestepEmbedder',
    'LabelEmbedder',
    'FinalLayer',
    
    # Transport
    'Transport',
    'Path',
    'Integrator',
    
    # Sampling
    'sample_ode',
    'sample_sde',
    
    # Losses
    'DispersiveLoss',
    'VICRegLoss',
    'VICRegDispersiveLoss',
    'CombinedLoss',
    'ContrastiveLoss',
    'InfoNCE',
    'SimCLRLoss',
    'L2Regularization',
    'L1Regularization',
    
    # Utils
    'CustomDataset',
    'ImageNetDataset',
    'CIFARDataset',
    'DiffusionDataset',
    'PairedDataset',
    'get_transforms',
    'create_augmentation_pipeline',
    'create_dataloader',
    'get_dataset_info',
    'split_dataset',
    
    # Configs
    'get_model_config',
    'get_training_config',
    'get_dispersive_loss_config',
    'SIT_CONFIGS',
    'TRAINING_CONFIGS',
    'DISPERSIVE_LOSS_CONFIGS',
]
