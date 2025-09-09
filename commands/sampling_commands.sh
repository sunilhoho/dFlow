#!/bin/bash
# Sampling Commands for dFlow
# This file contains example commands for sampling from trained models

echo "=== dFlow Sampling Commands ==="

# Basic sampling commands
echo "1. Basic Sampling Commands:"
echo "python main.py sample model=sit_xl_2 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_b_2 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_l_4 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Sampling with different methods
echo "2. Sampling with Different Methods:"
echo "python main.py sample model=sit_xl_2 sampling=ode_default checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_xl_2 sampling=sde_default checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_xl_2 sampling=ddim checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Sampling with parameter overrides
echo "3. Sampling with Parameter Overrides:"
echo "python main.py sample model=sit_xl_2 sampling.num_samples=16 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_b_2 sampling.num_steps=200 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_l_2 sampling.cfg_scale=2.0 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_xl_2 sampling.image_size=512 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Sampling with custom output directory
echo "4. Sampling with Custom Output Directory:"
echo "python main.py sample model=sit_xl_2 sample_dir=./my_samples checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_b_2 sample_dir=./experiment_1 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Multi-run sampling
echo "5. Multi-run Sampling:"
echo "python main.py sample --multirun model=sit_xl_2 sampling=ode_default,sde_default checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample --multirun model=sit_b_2,sit_l_2 sampling=ode_default checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Sampling for evaluation
echo "6. Sampling for Evaluation (Large Number of Samples):"
echo "python main.py sample model=sit_xl_2 sampling.num_samples=1000 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_b_2 sampling.num_samples=5000 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Sampling with different CFG scales
echo "7. Sampling with Different CFG Scales:"
echo "python main.py sample model=sit_xl_2 sampling.cfg_scale=1.0 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_xl_2 sampling.cfg_scale=1.5 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_xl_2 sampling.cfg_scale=2.0 checkpoint_path=./checkpoints/final_model.pt"
echo ""

# Sampling with different step counts
echo "8. Sampling with Different Step Counts:"
echo "python main.py sample model=sit_xl_2 sampling.num_steps=50 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_xl_2 sampling.num_steps=100 checkpoint_path=./checkpoints/final_model.pt"
echo "python main.py sample model=sit_xl_2 sampling.num_steps=200 checkpoint_path=./checkpoints/final_model.pt"
echo ""

echo "Note: Make sure to have trained models and update checkpoint paths before running these commands."
