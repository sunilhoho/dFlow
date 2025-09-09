# dFlow Replication Commands

This directory contains all the commands needed to replicate the experiments and results from the dFlow library.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -e .
   ```

2. **Run basic training:**
   ```bash
   python main.py train model=sit_xl_2
   ```

3. **Sample from trained model:**
   ```bash
   python main.py sample model=sit_xl_2 checkpoint_path=./checkpoints/final_model.pt
   ```

4. **Evaluate model:**
   ```bash
   python main.py evaluate model=sit_xl_2 checkpoint_path=./checkpoints/final_model.pt
   ```

## Command Files

- `training_commands.sh` - All training commands and examples
- `sampling_commands.sh` - All sampling commands and examples  
- `evaluation_commands.sh` - All evaluation commands and examples
- `replication_experiments.sh` - Complete replication experiments

## Replication Experiments

### Paper Results Replication

To replicate the results from the paper, run:

```bash
# SiT-B/2 with Dispersive Loss (80 epochs)
python main.py train model=sit_b_2 training=imagenet_256 loss=dispersive_default dataset=imagenet

# SiT-XL/2 with Dispersive Loss (80 epochs)  
python main.py train model=sit_xl_2 training=imagenet_256 loss=dispersive_default dataset=imagenet

# Evaluation
python main.py evaluate model=sit_b_2 checkpoint_path=./checkpoints/final_model.pt
python main.py evaluate model=sit_xl_2 checkpoint_path=./checkpoints/final_model.pt
```

### Ablation Studies

```bash
# Test different loss functions
python main.py train --multirun model=sit_b_2 loss=mse,dispersive_default,dispersive_strong

# Test different model sizes
python main.py train --multirun model=sit_s_2,sit_b_2,sit_l_2,sit_xl_2

# Test different schedulers
python main.py train --multirun model=sit_b_2 scheduler=cosine,step,exponential,plateau
```

### Multi-GPU Training

```bash
# Single GPU
python main.py train model=sit_xl_2

# Multi-GPU (if available)
torchrun --nproc_per_node=4 main.py train model=sit_xl_2
```

## Configuration

All configurations are in the `configs/` directory. You can override any parameter:

```bash
# Override specific parameters
python main.py train model=sit_xl_2 loss.lambda_disp=0.5 training.batch_size=64

# Use different datasets
python main.py train model=sit_b_2 dataset=cifar10 training=cifar10

# Use different sampling methods
python main.py sample model=sit_xl_2 sampling=sde_default checkpoint_path=./checkpoints/final_model.pt
```

## Output Structure

```
outputs/
├── 2024-01-01/
│   ├── 12-34-56/           # Timestamp
│   │   ├── .hydra/         # Configuration files
│   │   ├── checkpoints/    # Model checkpoints
│   │   ├── samples/        # Generated samples
│   │   └── evaluation/     # Evaluation results
│   └── ...
```

## Troubleshooting

1. **CUDA out of memory:** Reduce batch size in training config
2. **Configuration not found:** Check configs directory structure
3. **Checkpoint not found:** Update checkpoint paths in commands
4. **Dataset not found:** Update dataset paths in config files

## Advanced Usage

See individual command files for detailed examples:
- `./training_commands.sh` - Training examples
- `./sampling_commands.sh` - Sampling examples
- `./evaluation_commands.sh` - Evaluation examples
