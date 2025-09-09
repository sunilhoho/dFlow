# dFlow Configuration System

This document describes the comprehensive configuration system for dFlow (Dispersive Flow), built on top of Hydra for flexible and reproducible experiment management.

## Overview

The dFlow configuration system provides:
- **Modular Configuration**: Separate configs for models, datasets, losses, training, etc.
- **Hydra Integration**: Full Hydra support with overrides and composition
- **Environment Management**: Automatic environment setup and reproducibility
- **WandB Integration**: Seamless experiment tracking and logging
- **Flexible Overrides**: Easy parameter modification via command line

## Configuration Structure

```
configs/
├── config.yaml          # Main configuration file
├── base.yaml            # Base settings and defaults
├── setup.yaml           # Environment and system setup
├── launcher/            # Launcher configurations
│   └── local.yaml
├── models/              # Model configurations
│   ├── sit_s_2.yaml
│   ├── sit_b_2.yaml
│   ├── sit_l_2.yaml
│   └── sit_xl_2.yaml
├── datasets/            # Dataset configurations
│   ├── imagenet.yaml
│   ├── imagenet_512.yaml
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   └── custom.yaml
├── loss/                # Loss function configurations
│   ├── mse.yaml
│   ├── l1.yaml
│   ├── dispersive_default.yaml
│   ├── vicreg.yaml
│   ├── vicreg_dispersive.yaml
│   └── multi_component.yaml
├── training/            # Training configurations
│   ├── imagenet_256.yaml
│   ├── imagenet_512.yaml
│   ├── cifar10.yaml
│   └── cifar100.yaml
├── scheduler/           # Learning rate scheduler configurations
│   ├── cosine.yaml
│   ├── linear_warmup.yaml
│   ├── step.yaml
│   ├── exponential.yaml
│   └── plateau.yaml
└── sampling/            # Sampling configurations
    ├── ode_default.yaml
    ├── sde_default.yaml
    └── ddim.yaml
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python main.py

# Override specific parameters
python main.py model=sit/sit_l_2 loss=vicreg data.batch_size=64

# Use different dataset and training config
python main.py dataset=cifar10 training=cifar10 loss=dispersive_default

# Override multiple parameters
python main.py \
    model=sit/sit_xl_2 \
    training=imagenet_512 \
    loss=vicreg_dispersive \
    data.batch_size=16 \
    training.max_epochs=1000 \
    dispersive_loss.lambda_disp=0.5
```

### Advanced Usage

```bash
# Use specific config file
python main.py --config-path=configs --config-name=config

# Override with YAML file
python main.py --config-path=configs --config-name=config \
    --config-path=my_configs --config-name=my_experiment

# Multirun (parameter sweep)
python main.py -m model=sit/sit_s_2,sit/sit_b_2,sit/sit_l_2 \
    loss=dispersive_default,vicreg \
    data.batch_size=32,64,128

# Override with environment variables
HYDRA_CONFIG_PATH=configs python main.py model=sit/sit_xl_2
```

## Configuration Components

### Main Configuration (`config.yaml`)

The main configuration file defines the overall structure and defaults:

```yaml
defaults:
  - base
  - model: sit/sit_xl_2
  - training: imagenet_256
  - scheduler: cosine
  - loss: mse
  - dataset: imagenet
  - sampling: ode_default
  - _self_

# Project information
project_name: dflow
project_version: 0.1.0

# Experiment settings
experiment:
  name: null
  tags: []
  notes: ""
  group: null

# ... other settings
```

### Base Configuration (`base.yaml`)

Contains fundamental settings common across all experiments:

```yaml
# Project metadata
project:
  name: dflow
  version: 0.1.0

# Paths and directories
paths:
  data_root: /path/to/data
  output_root: ./outputs
  checkpoint_dir: ./checkpoints

# Model configuration
model:
  type: "sit"
  architecture: "sit_xl_2"

# Training configuration
training:
  max_epochs: 1000
  gradient_clip_val: 1.0

# ... other settings
```

### Setup Configuration (`setup.yaml`)

Handles environment setup, logging, and system configuration:

```yaml
# WandB configuration
wandb:
  project: dflow
  mode: run
  resume: allow

# Environment configuration
environment:
  seed: 42
  cuda:
    visible_devices: "0"
    allow_tf32: true

# Logging configuration
logging:
  console:
    level: INFO
  file:
    enabled: true
    level: DEBUG

# ... other settings
```

## Model Configurations

### SiT Model Configurations

```yaml
# configs/models/sit_xl_2.yaml
_target_: dflow.models.SiT
hidden_size: 1152
depth: 28
num_heads: 16
mlp_ratio: 4.0
patch_size: 2
num_classes: 1000
learn_sigma: true
```

## Dataset Configurations

### ImageNet Configuration

```yaml
# configs/dataset/imagenet.yaml
_target_: dflow.utils.ImageNetDataset
root: ${paths.data_root}/imagenet
split: train
image_size: 256
normalize: true
augmentation: true
```

### CIFAR-10 Configuration

```yaml
# configs/dataset/cifar10.yaml
_target_: dflow.utils.CIFARDataset
root: ${paths.data_root}/cifar10
dataset_name: CIFAR10
train: true
image_size: 32
normalize: true
augmentation: true
```

## Loss Configurations

### Dispersive Loss

```yaml
# configs/loss/dispersive_default.yaml
_target_: dflow.losses.DispersiveLoss
lambda_disp: 0.25
tau: 1.0
```

### VICReg Loss

```yaml
# configs/loss/vicreg.yaml
_target_: dflow.losses.VICRegLoss
lambda_inv: 25.0
lambda_var: 25.0
lambda_cov: 1.0
aggregation: mean
```

### Multi-Component Loss

```yaml
# configs/loss/multi_component.yaml
_target_: dflow.losses.MultiComponentLoss
components:
  main:
    _target_: torch.nn.MSELoss
  dispersive:
    _target_: dflow.losses.DispersiveLoss
    lambda_disp: 0.25
  vicreg:
    _target_: dflow.losses.VICRegLoss
    lambda_inv: 25.0
weights:
  main: 1.0
  dispersive: 0.25
  vicreg: 0.1
```

## Training Configurations

### ImageNet Training

```yaml
# configs/training/imagenet_256.yaml
_target_: dflow.training.train_with_dispersive_loss
max_epochs: 1000
learning_rate: 1e-4
batch_size: 32
image_size: 256
gradient_clip_val: 1.0
```

## Scheduler Configurations

### Cosine Annealing

```yaml
# configs/scheduler/cosine.yaml
_target_: torch.optim.lr_scheduler.CosineAnnealingLR
T_max: ${training.max_epochs}
eta_min: 1e-6
```

## Sampling Configurations

### ODE Sampling

```yaml
# configs/sampling/ode_default.yaml
_target_: dflow.sampling.sample_ode
method: euler
num_steps: 50
rtol: 1e-5
atol: 1e-5
```

## Environment Variables

The configuration system supports environment variables:

```bash
# Set data path
export DATA_ROOT=/path/to/your/data
python main.py

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1
python main.py device.devices=2

# Set WandB project
export WANDB_PROJECT=my_experiment
python main.py
```

## Custom Configurations

### Creating Custom Configs

1. Create a new YAML file in the appropriate directory
2. Define the configuration parameters
3. Use the config in your experiments

```yaml
# configs/loss/my_custom_loss.yaml
_target_: my_module.MyCustomLoss
param1: value1
param2: value2
```

### Using Custom Configs

```bash
# Use custom loss
python main.py loss=my_custom_loss

# Override custom loss parameters
python main.py loss=my_custom_loss loss.param1=new_value
```

## Best Practices

### 1. Use Meaningful Names

```yaml
# Good
configs/training/imagenet_256_dispersive.yaml

# Avoid
configs/training/config1.yaml
```

### 2. Group Related Parameters

```yaml
# Good
dispersive_loss:
  enabled: true
  lambda_disp: 0.25
  tau: 1.0

# Avoid
dispersive_enabled: true
dispersive_lambda: 0.25
dispersive_tau: 1.0
```

### 3. Use Defaults and Overrides

```yaml
# In base config
defaults:
  - model: sit/sit_xl_2
  - loss: mse

# Override when needed
python main.py loss=vicreg
```

### 4. Document Your Configs

```yaml
# Add comments to explain parameters
dispersive_loss:
  lambda_disp: 0.25  # Weight for dispersive loss component
  tau: 1.0           # Temperature parameter for softmax
```

## Troubleshooting

### Common Issues

1. **Config not found**: Check the path and file name
2. **Override not working**: Ensure the parameter path is correct
3. **Type errors**: Check parameter types in the config files

### Debug Mode

```bash
# Enable debug logging
python main.py hydra.verbose=true

# Print configuration
python main.py --cfg job
```

## Examples

See `examples/config_examples.py` for comprehensive examples of using the configuration system.

## Integration with Scripts

The configuration system integrates with all dFlow scripts:

```bash
# Training
python main.py experiment.job_type=train

# Sampling
python main.py experiment.job_type=sample

# Evaluation
python main.py experiment.job_type=evaluate
```

This configuration system provides a powerful and flexible way to manage experiments in dFlow while maintaining reproducibility and ease of use.
