"""
Example training scripts using Hydra configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_training_example():
    """Run training with different SiT model configurations."""
    
    examples = [
        {
            "name": "SiT-S/2 Training",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_s_2",
                "training=imagenet_256",
                "data_path=/path/to/imagenet/train",
                "experiment_name=sit_s_2_experiment"
            ]
        },
        {
            "name": "SiT-XL/2 Training with Dispersive Loss",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_xl_2",
                "training=imagenet_256",
                "training.dispersive_loss.enabled=true",
                "data_path=/path/to/imagenet/train",
                "experiment_name=sit_xl_2_disp_experiment"
            ]
        },
        {
            "name": "SiT-L/4 Training for 512x512",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_l_4",
                "training=imagenet_512",
                "data_path=/path/to/imagenet/train",
                "experiment_name=sit_l_4_512_experiment"
            ]
        }
    ]
    
    print("Available training examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Command: {' '.join(example['cmd'])}")
        print()
    
    # Note: These are example commands - you would run them manually
    print("To run an example, copy the command and execute it in the terminal.")
    print("Make sure to update the data_path to your actual dataset location.")


def run_sampling_example():
    """Run sampling with different configurations."""
    
    examples = [
        {
            "name": "ODE Sampling with SiT-XL/2",
            "cmd": [
                "python", "sample_hydra.py",
                "model=sit_xl_2",
                "sampling=ode_default",
                "sampling.num_samples=8",
                "checkpoint_path=/path/to/checkpoint.pt"
            ]
        },
        {
            "name": "SDE Sampling with SiT-B/2",
            "cmd": [
                "python", "sample_hydra.py",
                "model=sit_b_2",
                "sampling=sde_default",
                "sampling.num_samples=16",
                "checkpoint_path=/path/to/checkpoint.pt"
            ]
        },
        {
            "name": "DDIM Sampling with CFG",
            "cmd": [
                "python", "sample_hydra.py",
                "model=sit_l_2",
                "sampling=ddim",
                "sampling.cfg_scale=1.5",
                "sampling.num_samples=4",
                "checkpoint_path=/path/to/checkpoint.pt"
            ]
        }
    ]
    
    print("Available sampling examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Command: {' '.join(example['cmd'])}")
        print()
    
    print("To run an example, copy the command and execute it in the terminal.")
    print("Make sure to update the checkpoint_path to your actual model checkpoint.")


if __name__ == "__main__":
    print("=== dFlow Hydra Examples ===\n")
    
    print("1. Training Examples:")
    print("-" * 30)
    run_training_example()
    
    print("\n2. Sampling Examples:")
    print("-" * 30)
    run_sampling_example()
    
    print("\n3. Configuration Override Examples:")
    print("-" * 40)
    print("Override specific parameters:")
    print("python train_hydra.py model=sit_xl_2 training.batch_size=64 training.learning_rate=2e-4")
    print("python sample_hydra.py model=sit_b_2 sampling.num_samples=32 sampling.cfg_scale=2.0")
    
    print("\n4. Multi-run Examples:")
    print("-" * 25)
    print("Run multiple experiments:")
    print("python train_hydra.py --multirun model=sit_s_2,sit_b_2,sit_l_2 training.learning_rate=1e-4,2e-4")
    print("python sample_hydra.py --multirun model=sit_xl_2 sampling=ode_default,sde_default")
