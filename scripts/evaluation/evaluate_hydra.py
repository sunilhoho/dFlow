"""
Hydra-compatible evaluation script for dFlow.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dflow.models import SiT
from dflow.sampling import sample_ode, sample_sde


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function with Hydra configuration."""
    
    # Print configuration
    print("Evaluation Configuration:")
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
    else:
        print("Warning: No checkpoint provided, using random weights")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    output_dir = Path(cfg.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluation metrics
    metrics = {}
    
    # Generate samples for evaluation
    print("Generating samples for evaluation...")
    num_eval_samples = getattr(cfg, 'num_eval_samples', 1000)
    shape = (num_eval_samples, 3, cfg.sampling.image_size, cfg.sampling.image_size)
    
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
    
    # Convert to [0, 1] range for metrics
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    # Load real data for comparison (if available)
    real_data = None
    if hasattr(cfg, 'dataset') and cfg.dataset:
        try:
            dataset = hydra.utils.instantiate(cfg.dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.dataset.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
                pin_memory=cfg.dataset.pin_memory
            )
            
            # Collect real data
            real_data_list = []
            for i, batch in enumerate(dataloader):
                if i * cfg.dataset.batch_size >= num_eval_samples:
                    break
                if isinstance(batch, (list, tuple)):
                    x, _ = batch
                else:
                    x = batch
                real_data_list.append(x)
            real_data = torch.cat(real_data_list, dim=0)[:num_eval_samples]
            real_data = real_data.to(device)
            print(f"Loaded {len(real_data)} real samples for comparison")
        except Exception as e:
            print(f"Warning: Could not load real data: {e}")
            real_data = None
    
    # Compute metrics
    print("Computing evaluation metrics...")
    
    # FID (Fréchet Inception Distance)
    if real_data is not None:
        try:
            fid = FrechetInceptionDistance(feature=2048, normalize=True)
            fid.update(real_data, real=True)
            fid.update(samples, real=False)
            fid_score = fid.compute()
            metrics['fid'] = float(fid_score)
            print(f"FID: {fid_score:.4f}")
        except Exception as e:
            print(f"Warning: Could not compute FID: {e}")
            metrics['fid'] = None
    
    # Inception Score
    try:
        inception = InceptionScore()
        inception.update(samples)
        is_mean, is_std = inception.compute()
        metrics['inception_score_mean'] = float(is_mean)
        metrics['inception_score_std'] = float(is_std)
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        print(f"Warning: Could not compute Inception Score: {e}")
        metrics['inception_score_mean'] = None
        metrics['inception_score_std'] = None
    
    # LPIPS (if available)
    try:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
        lpips_scores = []
        for i in range(0, len(samples), 10):  # Sample every 10th image
            batch = samples[i:i+10]
            if len(batch) >= 2:
                score = lpips(batch[:len(batch)//2], batch[len(batch)//2:len(batch)])
                lpips_scores.append(score)
        
        if lpips_scores:
            avg_lpips = torch.stack(lpips_scores).mean()
            metrics['lpips'] = float(avg_lpips)
            print(f"LPIPS: {avg_lpips:.4f}")
        else:
            metrics['lpips'] = None
    except Exception as e:
        print(f"Warning: Could not compute LPIPS: {e}")
        metrics['lpips'] = None
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save sample images
    if cfg.sampling.save_images:
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(samples[:100]):  # Save first 100 samples
            img_array = (sample.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img_path = sample_dir / f"sample_{i:04d}.png"
            img.save(img_path)
        
        print(f"Sample images saved to: {sample_dir}")
    
    # Create evaluation summary
    summary = {
        "model": cfg.model._target_,
        "checkpoint": getattr(cfg, 'checkpoint_path', 'random_weights'),
        "num_samples": num_eval_samples,
        "sampling_method": cfg.sampling.method,
        "sampling_steps": cfg.sampling.num_steps,
        "cfg_scale": cfg.sampling.cfg_scale,
        "metrics": metrics
    }
    
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Evaluation completed! Results saved to: {output_dir}")
    print("\nEvaluation Summary:")
    print(f"Model: {cfg.model._target_}")
    print(f"Checkpoint: {getattr(cfg, 'checkpoint_path', 'random_weights')}")
    print(f"Number of samples: {num_eval_samples}")
    print(f"Sampling method: {cfg.sampling.method}")
    print(f"Sampling steps: {cfg.sampling.num_steps}")
    print(f"CFG scale: {cfg.sampling.cfg_scale}")
    print("\nMetrics:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
