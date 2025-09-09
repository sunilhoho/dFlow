#!/bin/bash
# Evaluation Commands for dFlow
# This file contains example commands for evaluating trained models

echo "=== dFlow Evaluation Commands ==="

# Basic evaluation commands
echo "1. Basic Evaluation Commands:"
echo "python main.py evaluate model=sit_xl_2 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_b_2 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_l_4 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Evaluation with different sampling methods
echo "2. Evaluation with Different Sampling Methods:"
echo "python main.py evaluate model=sit_xl_2 sampling=ode_default checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_xl_2 sampling=sde_default checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_xl_2 sampling=ddim checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Evaluation with different number of samples
echo "3. Evaluation with Different Number of Samples:"
echo "python main.py evaluate model=sit_xl_2 num_eval_samples=1000 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_b_2 num_eval_samples=5000 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_l_2 num_eval_samples=10000 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Evaluation with different CFG scales
echo "4. Evaluation with Different CFG Scales:"
echo "python main.py evaluate model=sit_xl_2 sampling.cfg_scale=1.0 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_xl_2 sampling.cfg_scale=1.5 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_xl_2 sampling.cfg_scale=2.0 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Evaluation with different datasets
echo "5. Evaluation with Different Datasets:"
echo "python main.py evaluate model=sit_b_2 dataset=cifar10 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_b_2 dataset=cifar100 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_xl_2 dataset=imagenet checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Multi-run evaluation
echo "6. Multi-run Evaluation:"
echo "python main.py evaluate --multirun model=sit_xl_2 sampling=ode_default,sde_default checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate --multirun model=sit_b_2,sit_l_2 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Evaluation with custom output directory
echo "7. Evaluation with Custom Output Directory:"
echo "python main.py evaluate model=sit_xl_2 output_dir=./evaluation_results checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Evaluation of different checkpoints
echo "8. Evaluation of Different Checkpoints:"
echo "python main.py evaluate model=sit_xl_2 checkpoint_path=./checkpoints/checkpoint_epoch_20.pt"
echo "python main.py evaluate model=sit_xl_2 checkpoint_path=./checkpoints/checkpoint_epoch_40.pt"
echo "python main.py evaluate model=sit_xl_2 checkpoint_path=./checkpoints/checkpoint_epoch_60.pt"
echo "python main.py evaluate model=sit_xl_2 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Evaluation with different image sizes
echo "9. Evaluation with Different Image Sizes:"
echo "python main.py evaluate model=sit_xl_2 sampling.image_size=256 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py evaluate model=sit_xl_2 sampling.image_size=512 checkpoint_path=./checkpoints/final_model.pt"
echo ""

echo "Note: Make sure to have trained models and update checkpoint paths before running these commands."
echo "Evaluation will compute FID, Inception Score, and LPIPS metrics."
