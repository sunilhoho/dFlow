#!/bin/bash
# Complete Replication Experiments for dFlow
# This script runs all the experiments to replicate the paper results

echo "=== dFlow Complete Replication Experiments ==="
echo "This script will run all experiments to replicate the paper results."
echo "Make sure you have sufficient compute resources and time."
echo ""

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export WANDB_ENABLED=false  # Set to true if you want wandb logging

# Create output directories
mkdir -p ./outputs/replication
mkdir -p ./checkpoints/replication

echo "Starting replication experiments..."
echo ""

# Experiment 1: SiT-S/2 with different losses
echo "=== Experiment 1: SiT-S/2 Loss Ablation ==="
echo "Training SiT-S/2 with different loss functions..."
python main.py train --multirun \
    model=sit_s_2 \
    loss=mse,dispersive_default,dispersive_strong \
    training=cifar10 \
    dataset=cifar10 \
    output_dir=./outputs/replication/sit_s_2_loss_ablation

echo ""

# Experiment 2: SiT-B/2 with different schedulers
echo "=== Experiment 2: SiT-B/2 Scheduler Ablation ==="
echo "Training SiT-B/2 with different schedulers..."
python main.py train --multirun \
    model=sit_b_2 \
    scheduler=cosine,step,exponential,plateau \
    training=cifar10 \
    dataset=cifar10 \
    output_dir=./outputs/replication/sit_b_2_scheduler_ablation

echo ""

# Experiment 3: Model size comparison
echo "=== Experiment 3: Model Size Comparison ==="
echo "Training different model sizes..."
python main.py train --multirun \
    model=sit_s_2,sit_b_2,sit_l_2 \
    loss=dispersive_default \
    training=cifar10 \
    dataset=cifar10 \
    output_dir=./outputs/replication/model_size_comparison

echo ""

# Experiment 4: SiT-XL/2 with ImageNet (if you have the dataset)
echo "=== Experiment 4: SiT-XL/2 ImageNet Training ==="
echo "Training SiT-XL/2 on ImageNet (requires ImageNet dataset)..."
echo "Note: Update data_path in configs/dataset/imagenet.yaml before running"
# python main.py train \
#     model=sit_xl_2 \
#     training=imagenet_256 \
#     loss=dispersive_default \
#     dataset=imagenet \
#     output_dir=./outputs/replication/sit_xl_2_imagenet

echo ""

# Experiment 5: Sampling method comparison
echo "=== Experiment 5: Sampling Method Comparison ==="
echo "Sampling with different methods..."
python main.py sample --multirun \
    model=sit_b_2 \
    sampling=ode_default,sde_default,ddim \
    checkpoint_path=./checkpoints/replication/final_model.pt \
    output_dir=./outputs/replication/sampling_comparison

echo ""

# Experiment 6: CFG scale ablation
echo "=== Experiment 6: CFG Scale Ablation ==="
echo "Sampling with different CFG scales..."
python main.py sample --multirun \
    model=sit_l_2 \
    sampling.cfg_scale=1.0,1.5,2.0,2.5 \
    checkpoint_path=./checkpoints/replication/final_model.pt \
    output_dir=./outputs/replication/cfg_scale_ablation

echo ""

# Experiment 7: Evaluation of all models
echo "=== Experiment 7: Model Evaluation ==="
echo "Evaluating all trained models..."
python main.py evaluate --multirun \
    model=sit_s_2,sit_b_2,sit_l_2 \
    checkpoint_path=./checkpoints/replication/final_model.pt \
    output_dir=./outputs/replication/evaluation

echo ""

# Experiment 8: Multi-run with different hyperparameters
echo "=== Experiment 8: Hyperparameter Sweep ==="
echo "Running hyperparameter sweep..."
python main.py train --multirun \
    model=sit_b_2 \
    loss=dispersive_default \
    loss.lambda_disp=0.1,0.25,0.5,1.0 \
    loss.tau=0.5,1.0,2.0 \
    training=cifar10 \
    dataset=cifar10 \
    output_dir=./outputs/replication/hyperparameter_sweep

echo ""

echo "=== Replication Experiments Complete ==="
echo "All experiments have been completed."
echo "Results are saved in ./outputs/replication/"
echo ""
echo "To view results:"
echo "  ls -la ./outputs/replication/"
echo "  cat ./outputs/replication/*/evaluation_summary.json"
echo ""
echo "To generate sample images:"
echo "  python main.py sample model=sit_b_2 checkpoint_path=./checkpoints/replication/final_model.pt"
