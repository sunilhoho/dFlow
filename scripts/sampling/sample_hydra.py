"""
Hydra-compatible sampling script for dFlow.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from dflow.models import SiT
from dflow.sampling import sample_ode, sample_sde


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main sampling function with Hydra configuration."""
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    
    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    
    # Load checkpoint if provided
    if hasattr(cfg, 'checkpoint_path') and cfg.checkpoint_path:
        print(f"Loading checkpoint: {cfg.checkpoint_path}")
        checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully!")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    output_dir = Path(cfg.sample_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    print(f"Generating {cfg.sampling.num_samples} samples...")
    
    shape = (cfg.sampling.num_samples, 3, cfg.sampling.image_size, cfg.sampling.image_size)
    
    with torch.no_grad():
        if cfg.sampling.method == "ode":
            samples = sample_ode(
                model,
                shape,
                num_steps=cfg.sampling.num_steps,
                device=device,
                cfg_scale=cfg.sampling.cfg_scale
            )
        elif cfg.sampling.method == "sde":
            samples = sample_sde(
                model,
                shape,
                num_steps=cfg.sampling.num_steps,
                device=device,
                cfg_scale=cfg.sampling.cfg_scale
            )
        else:
            raise ValueError(f"Unknown sampling method: {cfg.sampling.method}")
    
    # Convert to numpy and denormalize
    samples = samples.cpu().numpy()
    samples = (samples + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    samples = np.clip(samples, 0, 1)
    
    # Save images
    if cfg.sampling.save_images:
        for i, sample in enumerate(samples):
            # Convert to PIL Image
            img_array = (sample.transpose(1, 2, 0) * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            # Save image
            img_path = output_dir / f"sample_{i:04d}.{cfg.sampling.output_format}"
            img.save(img_path)
            print(f"Saved: {img_path}")
    
    # Save as npz if requested
    if cfg.sampling.save_npz:
        npz_path = output_dir / "samples.npz"
        np.savez(npz_path, samples=samples)
        print(f"Saved npz: {npz_path}")
    
    # Create a grid visualization
    if cfg.sampling.num_samples > 1:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(min(4, cfg.sampling.num_samples)):
            img_array = (samples[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            axes[i].imshow(img_array)
            axes[i].axis('off')
            axes[i].set_title(f'Sample {i+1}')
        
        # Hide unused subplots
        for i in range(4, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        grid_path = output_dir / "sample_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved grid: {grid_path}")
    
    print(f"Sampling completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
