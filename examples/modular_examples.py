"""
Example usage of modular Hydra configurations for dFlow.
"""

def show_training_examples():
    """Show training examples with modular configurations."""
    
    examples = [
        {
            "name": "SiT-XL/2 with Cosine Scheduler and Dispersive Loss",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_xl_2",
                "training=imagenet_256",
                "scheduler=cosine",
                "loss=dispersive_default",
                "dataset=imagenet"
            ]
        },
        {
            "name": "SiT-B/2 with Step Scheduler and MSE Loss",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_b_2",
                "training=cifar10",
                "scheduler=step",
                "loss=mse",
                "dataset=cifar10"
            ]
        },
        {
            "name": "SiT-L/4 with Strong Dispersive Loss and Plateau Scheduler",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_l_4",
                "training=imagenet_512",
                "scheduler=plateau",
                "loss=dispersive_strong",
                "dataset=imagenet_512"
            ]
        },
        {
            "name": "SiT-S/2 with Combined Loss and Exponential Scheduler",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_s_2",
                "training=cifar100",
                "scheduler=exponential",
                "loss=combined",
                "dataset=cifar100"
            ]
        }
    ]
    
    print("Available training examples with modular configurations:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Command: {' '.join(example['cmd'])}")
        print()


def show_parameter_override_examples():
    """Show parameter override examples."""
    
    examples = [
        {
            "name": "Override Loss Parameters",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_xl_2",
                "loss=dispersive_default",
                "loss.lambda_disp=0.5",
                "loss.tau=0.5"
            ]
        },
        {
            "name": "Override Scheduler Parameters",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_b_2",
                "scheduler=cosine",
                "scheduler.eta_min=1e-7",
                "scheduler.T_max=500000"
            ]
        },
        {
            "name": "Override Dataset Parameters",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_l_2",
                "dataset=imagenet",
                "dataset.batch_size=64",
                "dataset.num_workers=8"
            ]
        },
        {
            "name": "Override Training Parameters",
            "cmd": [
                "python", "train_hydra.py",
                "model=sit_xl_2",
                "training.batch_size=16",
                "training.learning_rate=2e-4",
                "training.num_epochs=100"
            ]
        }
    ]
    
    print("Parameter override examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Command: {' '.join(example['cmd'])}")
        print()


def show_multirun_examples():
    """Show multi-run experiment examples."""
    
    examples = [
        {
            "name": "Test Different Loss Functions",
            "cmd": [
                "python", "train_hydra.py", "--multirun",
                "model=sit_b_2",
                "loss=mse,dispersive_default,dispersive_strong"
            ]
        },
        {
            "name": "Test Different Schedulers",
            "cmd": [
                "python", "train_hydra.py", "--multirun",
                "model=sit_l_2",
                "scheduler=cosine,step,exponential,plateau"
            ]
        },
        {
            "name": "Test Different Models with Same Config",
            "cmd": [
                "python", "train_hydra.py", "--multirun",
                "model=sit_s_2,sit_b_2,sit_l_2",
                "loss=dispersive_default",
                "scheduler=cosine"
            ]
        },
        {
            "name": "Test Different Datasets",
            "cmd": [
                "python", "train_hydra.py", "--multirun",
                "model=sit_b_2",
                "dataset=cifar10,cifar100",
                "training=cifar10"
            ]
        }
    ]
    
    print("Multi-run experiment examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Command: {' '.join(example['cmd'])}")
        print()


def show_available_configs():
    """Show all available configurations."""
    
    configs = {
        "Models": [
            "sit_s_2", "sit_s_4", "sit_s_8",
            "sit_b_2", "sit_b_4", "sit_b_8", 
            "sit_l_2", "sit_l_4", "sit_l_8",
            "sit_xl_2", "sit_xl_4", "sit_xl_8"
        ],
        "Training": [
            "imagenet_256", "imagenet_512", "cifar10", "cifar100"
        ],
        "Schedulers": [
            "cosine", "linear_warmup", "step", "exponential", "plateau"
        ],
        "Loss Functions": [
            "mse", "l1", "huber", "dispersive_default", "dispersive_strong", "combined"
        ],
        "Datasets": [
            "imagenet", "imagenet_512", "cifar10", "cifar100", "custom"
        ],
        "Sampling": [
            "ode_default", "sde_default", "ddim"
        ]
    }
    
    print("Available configurations:")
    for category, config_list in configs.items():
        print(f"\n{category}:")
        for config in config_list:
            print(f"  - {config}")


if __name__ == "__main__":
    print("=== dFlow Modular Configuration Examples ===\n")
    
    print("1. Training Examples:")
    print("-" * 40)
    show_training_examples()
    
    print("\n2. Parameter Override Examples:")
    print("-" * 40)
    show_parameter_override_examples()
    
    print("\n3. Multi-run Experiment Examples:")
    print("-" * 40)
    show_multirun_examples()
    
    print("\n4. Available Configurations:")
    print("-" * 40)
    show_available_configs()
    
    print("\n5. Quick Reference:")
    print("-" * 20)
    print("Override any parameter using dot notation:")
    print("  python train_hydra.py model=sit_xl_2 loss.lambda_disp=0.5")
    print("  python train_hydra.py model=sit_b_2 scheduler.gamma=0.9")
    print("  python train_hydra.py model=sit_l_2 dataset.batch_size=64")
    
    print("\nUse --multirun for multiple experiments:")
    print("  python train_hydra.py --multirun model=sit_s_2,sit_b_2 loss=mse,dispersive_default")
    
    print("\nView all available options:")
    print("  python train_hydra.py --config-path=configs --config-name=config --cfg job")
