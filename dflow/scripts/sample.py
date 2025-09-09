"""
Command-line sampling script for dFlow.
"""

import argparse
import torch
from dflow.models import SiT
from dflow.configs import get_model_config
from dflow.sampling import sample_ode, sample_sde

def main():
    parser = argparse.ArgumentParser(description='Sample from SiT model')
    parser.add_argument('method', choices=['ODE', 'SDE'], help='Sampling method')
    parser.add_argument('--model', type=str, default='SiT-XL/2', help='Model variant')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image-size', type=int, default=256, help='Image size')
    parser.add_argument('--num-samples', type=int, default=4, help='Number of samples')
    parser.add_argument('--num-steps', type=int, default=100, help='Number of sampling steps')
    parser.add_argument('--cfg-scale', type=float, default=1.0, help='Classifier-free guidance scale')
    parser.add_argument('--device', type=str, default='cuda', help='Device to sample on')
    parser.add_argument('--output', type=str, default='samples.png', help='Output file path')
    
    args = parser.parse_args()
    
    # Get model configuration
    model_config = get_model_config(args.model)
    
    # Create model
    model = SiT(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)
    model.eval()
    
    # Sample images
    shape = (args.num_samples, 3, args.image_size, args.image_size)
    
    if args.method == 'ODE':
        images = sample_ode(
            model, 
            shape, 
            num_steps=args.num_steps,
            device=args.device,
            cfg_scale=args.cfg_scale
        )
    else:  # SDE
        images = sample_sde(
            model,
            shape,
            num_steps=args.num_steps,
            device=args.device,
            cfg_scale=args.cfg_scale
        )
    
    # Save images (you would need to implement image saving here)
    print(f"Generated {args.num_samples} images using {args.method} sampling")
    print(f"Images shape: {images.shape}")

if __name__ == '__main__':
    main()
