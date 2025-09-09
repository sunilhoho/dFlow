"""
Examples demonstrating dFlow configuration system.

This script shows how to use the various configuration files
and Hydra integration in the dFlow library.
"""

import os
import sys
from pathlib import Path

# Add dflow to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
from dflow.utils import create_dataloader_from_config


def example_basic_config():
    """Example of basic configuration usage."""
    print("=== Basic Configuration Example ===")
    
    # Load configuration
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
    
    print("Configuration loaded successfully!")
    print(f"Project: {cfg.project_name}")
    print(f"Model: {cfg.model._target_}")
    print(f"Training epochs: {cfg.training.max_epochs}")
    print(f"Batch size: {cfg.data.batch_size}")
    print(f"Dispersive loss enabled: {cfg.dispersive_loss.enabled}")
    print()


def example_config_override():
    """Example of configuration overrides."""
    print("=== Configuration Override Example ===")
    
    # Load configuration with overrides
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "model=sit/sit_l_2",
                "training=imagenet_512",
                "loss=vicreg",
                "data.batch_size=64",
                "training.max_epochs=500",
                "dispersive_loss.lambda_disp=0.5"
            ]
        )
    
    print("Configuration with overrides loaded!")
    print(f"Model: {cfg.model._target_}")
    print(f"Training config: {cfg.training._target_}")
    print(f"Loss: {cfg.loss._target_}")
    print(f"Batch size: {cfg.data.batch_size}")
    print(f"Max epochs: {cfg.training.max_epochs}")
    print(f"Dispersive lambda: {cfg.dispersive_loss.lambda_disp}")
    print()


def example_model_configs():
    """Example of different model configurations."""
    print("=== Model Configuration Examples ===")
    
    model_configs = [
        "sit/sit_s_2",
        "sit/sit_b_2", 
        "sit/sit_l_2",
        "sit/sit_xl_2"
    ]
    
    for model_name in model_configs:
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[f"model={model_name}"]
            )
        
        print(f"Model: {model_name}")
        print(f"  Target: {cfg.model._target_}")
        print(f"  Hidden size: {cfg.model.hidden_size}")
        print(f"  Depth: {cfg.model.depth}")
        print(f"  Num heads: {cfg.model.num_heads}")
        print()


def example_loss_configs():
    """Example of different loss configurations."""
    print("=== Loss Configuration Examples ===")
    
    loss_configs = [
        "mse",
        "l1",
        "huber",
        "dispersive_default",
        "vicreg",
        "vicreg_dispersive",
        "multi_component"
    ]
    
    for loss_name in loss_configs:
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[f"loss={loss_name}"]
            )
        
        print(f"Loss: {loss_name}")
        print(f"  Target: {cfg.loss._target_}")
        if hasattr(cfg.loss, 'reduction'):
            print(f"  Reduction: {cfg.loss.reduction}")
        if hasattr(cfg.loss, 'lambda_disp'):
            print(f"  Lambda disp: {cfg.loss.lambda_disp}")
        if hasattr(cfg.loss, 'lambda_inv'):
            print(f"  Lambda inv: {cfg.loss.lambda_inv}")
        print()


def example_dataset_configs():
    """Example of different dataset configurations."""
    print("=== Dataset Configuration Examples ===")
    
    dataset_configs = [
        "imagenet",
        "imagenet_512",
        "cifar10",
        "cifar100",
        "custom"
    ]
    
    for dataset_name in dataset_configs:
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[f"dataset={dataset_name}"]
            )
        
        print(f"Dataset: {dataset_name}")
        print(f"  Target: {cfg.dataset._target_}")
        print(f"  Data path: {cfg.dataset.data_path}")
        print(f"  Image size: {cfg.dataset.image_size}")
        print(f"  Batch size: {cfg.dataset.batch_size}")
        print()


def example_training_configs():
    """Example of different training configurations."""
    print("=== Training Configuration Examples ===")
    
    training_configs = [
        "imagenet_256",
        "imagenet_512",
        "cifar10",
        "cifar100"
    ]
    
    for training_name in training_configs:
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[f"training={training_name}"]
            )
        
        print(f"Training: {training_name}")
        print(f"  Target: {cfg.training._target_}")
        print(f"  Max epochs: {cfg.training.max_epochs}")
        print(f"  Learning rate: {cfg.training.learning_rate}")
        print(f"  Batch size: {cfg.training.batch_size}")
        print()


def example_scheduler_configs():
    """Example of different scheduler configurations."""
    print("=== Scheduler Configuration Examples ===")
    
    scheduler_configs = [
        "cosine",
        "linear_warmup",
        "step",
        "exponential",
        "plateau"
    ]
    
    for scheduler_name in scheduler_configs:
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[f"scheduler={scheduler_name}"]
            )
        
        print(f"Scheduler: {scheduler_name}")
        print(f"  Target: {cfg.scheduler._target_}")
        if hasattr(cfg.scheduler, 'T_max'):
            print(f"  T_max: {cfg.scheduler.T_max}")
        if hasattr(cfg.scheduler, 'step_size'):
            print(f"  Step size: {cfg.scheduler.step_size}")
        if hasattr(cfg.scheduler, 'gamma'):
            print(f"  Gamma: {cfg.scheduler.gamma}")
        print()


def example_sampling_configs():
    """Example of different sampling configurations."""
    print("=== Sampling Configuration Examples ===")
    
    sampling_configs = [
        "ode_default",
        "sde_default",
        "ddim"
    ]
    
    for sampling_name in sampling_configs:
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[f"sampling={sampling_name}"]
            )
        
        print(f"Sampling: {sampling_name}")
        print(f"  Target: {cfg.sampling._target_}")
        print(f"  Method: {cfg.sampling.method}")
        print(f"  Num steps: {cfg.sampling.num_steps}")
        if hasattr(cfg.sampling, 'eta'):
            print(f"  Eta: {cfg.sampling.eta}")
        print()


def example_environment_config():
    """Example of environment configuration."""
    print("=== Environment Configuration Example ===")
    
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
    
    print("Environment settings:")
    print(f"  Seed: {cfg.reproducibility.seed}")
    print(f"  Deterministic: {cfg.reproducibility.deterministic}")
    print(f"  Device: {cfg.device.accelerator}")
    print(f"  Precision: {cfg.device.precision}")
    print(f"  CUDA devices: {cfg.environment.cuda.visible_devices}")
    print(f"  OMP threads: {cfg.environment.threads.omp_num_threads}")
    print()


def example_wandb_config():
    """Example of WandB configuration."""
    print("=== WandB Configuration Example ===")
    
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="config")
    
    print("WandB settings:")
    print(f"  Project: {cfg.wandb.project}")
    print(f"  Mode: {cfg.wandb.mode}")
    print(f"  Resume: {cfg.wandb.resume}")
    print(f"  Tags: {cfg.wandb.tags}")
    print()


def example_custom_config():
    """Example of creating a custom configuration."""
    print("=== Custom Configuration Example ===")
    
    # Create a custom configuration by composing multiple configs
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "model=sit/sit_xl_2",
                "training=imagenet_512",
                "loss=vicreg_dispersive",
                "dataset=imagenet_512",
                "scheduler=cosine",
                "sampling=ode_default",
                "data.batch_size=16",
                "training.max_epochs=1000",
                "dispersive_loss.lambda_disp=0.5",
                "vicreg_loss.lambda_inv=50.0",
                "device.precision=32",
                "wandb.project=my_experiment"
            ]
        )
    
    print("Custom configuration created!")
    print("Key settings:")
    print(f"  Model: {cfg.model._target_}")
    print(f"  Training: {cfg.training._target_}")
    print(f"  Loss: {cfg.loss._target_}")
    print(f"  Dataset: {cfg.dataset._target_}")
    print(f"  Batch size: {cfg.data.batch_size}")
    print(f"  Max epochs: {cfg.training.max_epochs}")
    print(f"  Dispersive lambda: {cfg.dispersive_loss.lambda_disp}")
    print(f"  VICReg lambda inv: {cfg.vicreg_loss.lambda_inv}")
    print(f"  Precision: {cfg.device.precision}")
    print(f"  WandB project: {cfg.wandb.project}")
    print()


def example_config_validation():
    """Example of configuration validation."""
    print("=== Configuration Validation Example ===")
    
    try:
        with hydra.initialize(config_path="../configs", version_base=None):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    "model=sit/sit_xl_2",
                    "loss=vicreg",
                    "data.batch_size=32"
                ]
            )
        
        print("Configuration validation passed!")
        print(f"Model: {cfg.model._target_}")
        print(f"Loss: {cfg.loss._target_}")
        print(f"Batch size: {cfg.data.batch_size}")
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
    
    print()


if __name__ == "__main__":
    print("dFlow Configuration Examples")
    print("=" * 50)
    print()
    
    example_basic_config()
    example_config_override()
    example_model_configs()
    example_loss_configs()
    example_dataset_configs()
    example_training_configs()
    example_scheduler_configs()
    example_sampling_configs()
    example_environment_config()
    example_wandb_config()
    example_custom_config()
    example_config_validation()
    
    print("All configuration examples completed!")
    print("\nTo run with Hydra, use:")
    print("  python main.py")
    print("  python main.py model=sit/sit_l_2 loss=vicreg data.batch_size=64")
    print("  python main.py --config-path=configs --config-name=config")
