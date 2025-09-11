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
import os

from dflow.models import SiT
from dflow.transport import Sampler
from dflow.utils import hydra as hydra_utils

@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main sampling function with Hydra configuration."""
    # import pdb; pdb.set_trace()
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    transport = hydra.utils.instantiate(cfg.transport)
    transport_sampler = Sampler(transport)
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    # Load checkpoint if provided
    if os.path.exists(os.path.join(cfg.output_dir, 'train', 'checkpoints', 'final.pt')):
        print(f"Loading checkpoint: {os.path.join(cfg.output_dir, 'train', 'checkpoints', 'final.pt')}")
        checkpoint = torch.load(
            os.path.join(cfg.output_dir, 'train', 'checkpoints', 'final.pt'),
            map_location=device
        )

        # We'll handle two versions: naive and ema
        model_variants = {
            "naive": checkpoint["model_state_dict"],
            "ema": checkpoint["ema_state_dict"],
        }

        for variant_name, state_dict in model_variants.items():
            print(f"\n=== Generating with {variant_name.upper()} weights ===")
            # Reload model each time to avoid weight mixing
            variant_model = hydra.utils.instantiate(cfg.model).to(device)
            variant_model.load_state_dict(state_dict)
            variant_model.eval()

            # Make subdir for this variant
            variant_outdir = os.path.join(cfg.output_dir, 'samples', variant_name, f'{cfg.sampling.num_steps}_{cfg.sampling.method}')
            os.makedirs(variant_outdir, exist_ok=True)

            # === Sampling ===
            ys = torch.randint(cfg.model.num_classes,
                               size=(cfg.sampling.num_samples,),
                               device=device)
            use_cfg = cfg.get('cfg_scale', 1.0) > 1.0

            n = ys.size(0)
            if cfg.dataset.image_size == 256:
                zs = torch.randn(n, 4, cfg.model.input_size, cfg.model.input_size, device=device)
            else:
                zs = torch.randn(n, 3, cfg.model.input_size, cfg.model.input_size, device=device)

            if use_cfg:
                zs = torch.cat([zs, zs], 0)
                y_null = torch.tensor([cfg.model.num_classes] * n, device=device)
                ys = torch.cat([ys, y_null], 0)
                sample_model_kwargs = dict(y=ys, cfg_scale=cfg.cfg_scale)
                model_fn = variant_model.forward_with_cfg
            else:
                sample_model_kwargs = dict(y=ys)
                model_fn = variant_model.forward

            with torch.no_grad():
                if cfg.sampling.method == "ode":
                    sample_fn = transport_sampler.sample_ode()
                elif cfg.sampling.method == "sde":
                    sample_fn = transport_sampler.sample_ode()
                else:
                    raise ValueError(f"Unknown sampling method: {cfg.sampling.method}")

                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]

            # Convert to numpy
            samples = samples.cpu().numpy()
            samples = (samples + 1) / 2
            samples = np.clip(samples, 0, 1)

            # Save individual images
            if cfg.sampling.save_images:
                for i, sample in enumerate(samples):
                    img_array = (sample.transpose(1, 2, 0) * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    img_path = os.path.join(variant_outdir, f"sample_{i:04d}.{cfg.sampling.output_format}")
                    img.save(img_path)
                print(f"Saved {len(samples)} images to {variant_outdir}")

            # Save npz
            if cfg.sampling.save_npz:
                npz_path = os.path.join(variant_outdir, "samples.npz")
                np.savez(npz_path, samples=samples)
                print(f"Saved npz: {npz_path}")

            # Save grid
            if cfg.sampling.num_samples > 1:
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                axes = axes.flatten()
                for i in range(min(4, cfg.sampling.num_samples)):
                    img_array = (samples[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    axes[i].imshow(img_array)
                    axes[i].axis('off')
                    axes[i].set_title(f'Sample {i+1}')
                for i in range(4, len(axes)):
                    axes[i].axis('off')
                plt.tight_layout()
                grid_path = os.path.join(variant_outdir, "sample_grid.png")
                plt.savefig(grid_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved grid: {grid_path}")

        print("\n✅ Sampling completed for both naive and EMA models!")

if __name__ == "__main__":
    main()
