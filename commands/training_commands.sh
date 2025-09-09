#!/bin/bash
# Training Commands for dFlow
# This file contains example commands for training different SiT models

echo "=== dFlow Training Commands ==="

# Basic training commands
echo "1. Basic Training Commands:"
echo "python main.py train model=sit_xl_2"
echo "python main.py train model=sit_b_2"
echo "python main.py train model=sit_l_4"
echo "python main.py train model=sit_s_2"
echo ""

# Training with specific configurations
echo "2. Training with Specific Configurations:"
echo "python main.py train model=sit_xl_2 scheduler=cosine loss=dispersive_default dataset=imagenet"
echo "python main.py train model=sit_b_2 scheduler=step loss=mse dataset=cifar10"
echo "python main.py train model=sit_l_4 scheduler=plateau loss=combined dataset=imagenet_512"
echo ""

# Training with parameter overrides
echo "3. Training with Parameter Overrides:"
echo "python main.py train model=sit_xl_2 loss.lambda_disp=0.5 loss.tau=0.5"
echo "python main.py train model=sit_b_2 scheduler.gamma=0.9 scheduler.step_size=50000"
echo "python main.py train model=sit_l_2 dataset.batch_size=64 dataset.num_workers=8"
echo "python main.py train model=sit_xl_2 training.learning_rate=2e-4 training.num_epochs=100"
echo ""

# Multi-run experiments
echo "4. Multi-run Experiments:"
echo "python main.py train --multirun model=sit_s_2,sit_b_2,sit_l_2"
echo "python main.py train --multirun loss=mse,dispersive_default,dispersive_strong"
echo "python main.py train --multirun scheduler=cosine,step,exponential,plateau"
echo "python main.py train --multirun dataset=cifar10,cifar100"
echo ""

# Training with different datasets
echo "5. Training with Different Datasets:"
echo "python main.py train model=sit_b_2 dataset=cifar10 training=cifar10"
echo "python main.py train model=sit_b_2 dataset=cifar100 training=cifar100"
echo "python main.py train model=sit_xl_2 dataset=imagenet training=imagenet_256"
echo "python main.py train model=sit_xl_2 dataset=imagenet_512 training=imagenet_512"
echo ""

# Training with wandb logging
echo "6. Training with Weights & Biases Logging:"
echo "export WANDB_KEY='your-key'"
echo "export WANDB_ENTITY='your-entity'"
echo "export WANDB_PROJECT='your-project'"
echo "python main.py train model=sit_xl_2 wandb.enabled=true"
echo ""

# Resume training
echo "7. Resume Training from Checkpoint:"
echo "python main.py train model=sit_xl_2 checkpoint_path=./checkpoints/checkpoint_epoch_50.pt"
echo ""

echo "Note: Make sure to update data paths in your configuration files before running these commands."
