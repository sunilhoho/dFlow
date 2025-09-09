# Hydra Configuration Guide for dFlow

This guide explains how to use Hydra configurations with the dFlow library for flexible and reproducible experiments.

## 🚀 Quick Start

### Installation

Make sure you have Hydra installed:

```bash
pip install hydra-core omegaconf
```

### Basic Usage

```bash
# Train with default configuration
python train_hydra.py

# Train with specific model
python train_hydra.py model=sit_xl_2

# Sample with specific configuration
python sample_hydra.py model=sit_b_2 sampling=ode_default
```

## 📁 Configuration Structure

```
configs/
├── config.yaml              # Main configuration file
├── models/
│   └── sit/                 # SiT model configurations
│       ├── sit_s_2.yaml     # SiT-S/2 model
│       ├── sit_s_4.yaml     # SiT-S/4 model
│       ├── sit_s_8.yaml     # SiT-S/8 model
│       ├── sit_b_2.yaml     # SiT-B/2 model
│       ├── sit_b_4.yaml     # SiT-B/4 model
│       ├── sit_b_8.yaml     # SiT-B/8 model
│       ├── sit_l_2.yaml     # SiT-L/2 model
│       ├── sit_l_4.yaml     # SiT-L/4 model
│       ├── sit_l_8.yaml     # SiT-L/8 model
│       ├── sit_xl_2.yaml    # SiT-XL/2 model
│       ├── sit_xl_4.yaml    # SiT-XL/4 model
│       └── sit_xl_8.yaml    # SiT-XL/8 model
├── training/
│   ├── imagenet_256.yaml    # Training config for 256x256
│   └── imagenet_512.yaml    # Training config for 512x512
└── sampling/
    ├── ode_default.yaml     # ODE sampling config
    ├── sde_default.yaml     # SDE sampling config
    └── ddim.yaml            # DDIM sampling config
```

## 🔧 Available Model Configurations

### SiT-S (Small) Models
- `sit_s_2`: 384 hidden size, 12 layers, 6 heads, patch size 2
- `sit_s_4`: 384 hidden size, 12 layers, 6 heads, patch size 4
- `sit_s_8`: 384 hidden size, 12 layers, 6 heads, patch size 8

### SiT-B (Base) Models
- `sit_b_2`: 768 hidden size, 12 layers, 12 heads, patch size 2
- `sit_b_4`: 768 hidden size, 12 layers, 12 heads, patch size 4
- `sit_b_8`: 768 hidden size, 12 layers, 12 heads, patch size 8

### SiT-L (Large) Models
- `sit_l_2`: 1024 hidden size, 24 layers, 16 heads, patch size 2
- `sit_l_4`: 1024 hidden size, 24 layers, 16 heads, patch size 4
- `sit_l_8`: 1024 hidden size, 24 layers, 16 heads, patch size 8

### SiT-XL (Extra Large) Models
- `sit_xl_2`: 1152 hidden size, 28 layers, 16 heads, patch size 2
- `sit_xl_4`: 1152 hidden size, 28 layers, 16 heads, patch size 4
- `sit_xl_8`: 1152 hidden size, 28 layers, 16 heads, patch size 8

## 🏃‍♂️ Training Examples

### Basic Training
```bash
# Train SiT-XL/2 on ImageNet 256x256
python train_hydra.py model=sit_xl_2 training=imagenet_256

# Train SiT-B/2 on ImageNet 512x512
python train_hydra.py model=sit_b_2 training=imagenet_512
```

### Training with Dispersive Loss
```bash
# Enable dispersive loss
python train_hydra.py model=sit_xl_2 training.dispersive_loss.enabled=true

# Custom dispersive loss parameters
python train_hydra.py model=sit_l_2 \
    training.dispersive_loss.lambda_disp=0.5 \
    training.dispersive_loss.tau=0.5
```

### Parameter Overrides
```bash
# Override batch size and learning rate
python train_hydra.py model=sit_xl_2 \
    training.batch_size=64 \
    training.learning_rate=2e-4

# Override data path and experiment name
python train_hydra.py model=sit_b_2 \
    data_path=/path/to/your/dataset \
    experiment_name=my_experiment
```

### Multi-run Experiments
```bash
# Train multiple models with different learning rates
python train_hydra.py --multirun \
    model=sit_s_2,sit_b_2,sit_l_2 \
    training.learning_rate=1e-4,2e-4,5e-5

# Train with different batch sizes
python train_hydra.py --multirun \
    model=sit_xl_2 \
    training.batch_size=16,32,64
```

## 🎨 Sampling Examples

### Basic Sampling
```bash
# ODE sampling with SiT-XL/2
python sample_hydra.py model=sit_xl_2 sampling=ode_default

# SDE sampling with SiT-B/2
python sample_hydra.py model=sit_b_2 sampling=sde_default

# DDIM sampling with SiT-L/2
python sample_hydra.py model=sit_l_2 sampling=ddim
```

### Sampling with Custom Parameters
```bash
# Generate more samples
python sample_hydra.py model=sit_xl_2 \
    sampling.num_samples=16 \
    sampling.num_steps=200

# Sampling with higher CFG scale
python sample_hydra.py model=sit_b_2 \
    sampling.cfg_scale=2.0 \
    sampling.num_samples=8

# Custom output directory
python sample_hydra.py model=sit_l_2 \
    sample_dir=./my_samples \
    sampling.num_samples=32
```

### Sampling with Checkpoint
```bash
# Load specific checkpoint
python sample_hydra.py model=sit_xl_2 \
    checkpoint_path=/path/to/checkpoint.pt \
    sampling=ode_default
```

## ⚙️ Configuration Details

### Model Configuration
Each model configuration includes:
- `input_size`: Input image size
- `patch_size`: Patch size for vision transformer
- `in_channels`: Number of input channels
- `hidden_size`: Hidden dimension size
- `depth`: Number of transformer layers
- `num_heads`: Number of attention heads
- `mlp_ratio`: MLP expansion ratio
- `class_dropout_prob`: Classifier dropout probability
- `num_classes`: Number of classes
- `learn_sigma`: Whether to learn noise variance

### Training Configuration
Training configurations include:
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `num_epochs`: Number of training epochs
- `warmup_steps`: Learning rate warmup steps
- `grad_clip_norm`: Gradient clipping norm
- `ema_decay`: EMA decay rate
- `dispersive_loss`: Dispersive loss configuration

### Sampling Configuration
Sampling configurations include:
- `method`: Sampling method (ode, sde, ddim)
- `num_steps`: Number of sampling steps
- `cfg_scale`: Classifier-free guidance scale
- `num_samples`: Number of samples to generate
- `image_size`: Output image size
- `save_images`: Whether to save images
- `save_npz`: Whether to save as npz file

## 🔍 Advanced Usage

### Custom Configuration Files
Create your own configuration files:

```yaml
# configs/models/sit/custom_model.yaml
# @package model
_target_: dflow.models.SiT

input_size: 64
patch_size: 4
in_channels: 3
hidden_size: 512
depth: 16
num_heads: 8
mlp_ratio: 4.0
class_dropout_prob: 0.1
num_classes: 1000
learn_sigma: true
```

Then use it:
```bash
python train_hydra.py model=custom_model
```

### Configuration Composition
Combine multiple configurations:

```bash
# Use custom model with default training
python train_hydra.py model=custom_model training=imagenet_256

# Override specific parameters
python train_hydra.py model=sit_xl_2 \
    training=imagenet_256 \
    training.batch_size=16 \
    training.learning_rate=5e-5
```

### Environment Variables
Set environment variables for sensitive data:

```bash
export WANDB_KEY="your-key"
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="your-project"

python train_hydra.py wandb.enabled=true
```

## 📊 Logging and Monitoring

### Weights & Biases Integration
```bash
# Enable wandb logging
python train_hydra.py wandb.enabled=true \
    wandb.project=my_project \
    wandb.entity=my_entity
```

### Output Directory Structure
Hydra automatically creates organized output directories:
```
outputs/
├── 2024-01-01/
│   ├── 12-34-56/           # Timestamp
│   │   ├── .hydra/         # Configuration files
│   │   ├── checkpoints/    # Model checkpoints
│   │   └── samples/        # Generated samples
│   └── ...
```

## 🐛 Troubleshooting

### Common Issues

1. **Configuration not found**:
   ```bash
   # Check available configurations
   python train_hydra.py --config-path=configs --config-name=config --cfg job
   ```

2. **Parameter override not working**:
   ```bash
   # Use dot notation for nested parameters
   python train_hydra.py training.dispersive_loss.enabled=true
   ```

3. **Checkpoint loading issues**:
   ```bash
   # Make sure checkpoint path is correct
   python sample_hydra.py checkpoint_path=/absolute/path/to/checkpoint.pt
   ```

### Debug Mode
```bash
# Run with debug information
python train_hydra.py --config-path=configs --config-name=config --cfg job --resolve
```

## 📚 Further Reading

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [dFlow Library Documentation](./README.md)
