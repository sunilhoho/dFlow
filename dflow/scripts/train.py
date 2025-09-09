"""
Command-line training script for dFlow.
"""

import argparse
import torch
from dflow.models import SiT
from dflow.configs import get_model_config, get_training_config
from dflow.training import train_with_dispersive_loss

def main():
    parser = argparse.ArgumentParser(description='Train SiT model with dispersive loss')
    parser.add_argument('--model', type=str, default='SiT-XL/2', help='Model variant')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--disp', action='store_true', help='Use dispersive loss')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    
    args = parser.parse_args()
    
    # Get configurations
    model_config = get_model_config(args.model)
    train_config = get_training_config('imagenet_256')
    train_config.update({
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
    })
    
    # Create model
    model = SiT(**model_config)
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        # Note: You would need to implement data loading here
        # avg_loss = train_with_dispersive_loss(model, dataloader, optimizer, args.device, train_config)
        print(f"Epoch {epoch+1}/{args.epochs}")

if __name__ == '__main__':
    main()
