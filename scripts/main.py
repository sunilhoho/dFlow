"""
Main entry point for dFlow (Dispersive Flow) experiments.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import random
from time import sleep
import subprocess
import csv
import numpy as np

import hydra
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add dflow to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)

def setup_environment(config: DictConfig):
    """Setup environment and seeds."""
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_model(config: DictConfig) -> nn.Module:
    """Create model from configuration."""
    model = hydra.utils.instantiate(config.model)
    return model


def create_dataloaders(config: DictConfig) -> DataLoader:
    """Create dataloader from configuration."""
    dataloader = hydra.utils.instantiate(config.dataset)
    return dataloader


def create_optimizer_and_scheduler(model: nn.Module, config: DictConfig) -> tuple:
    """Create optimizer and learning rate scheduler."""
    # Create optimizer
    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
    
    # Create scheduler
    scheduler = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)
    
    return optimizer, scheduler


def train_model(model: nn.Module, 
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                config: DictConfig) -> None:
    """Train the model."""
    device = next(model.parameters()).device
    
    # Create loss function from config
    loss_fn = hydra.utils.instantiate(config.loss)
    
    # Create dispersive loss from config if specified
    dispersive_loss = None
    if hasattr(config, 'dispersive_loss'):
        dispersive_loss = hydra.utils.instantiate(config.dispersive_loss)
    
    # Training loop
    model.train()
    for epoch in range(config.training.max_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            
            x = x.to(device)
            if y is not None:
                y = y.to(device)
            
            # Sample timesteps
            t = torch.randint(0, 1000, (x.shape[0],), device=device)
            
            # Forward pass
            if hasattr(model, 'forward_with_activations'):
                x_pred, activations = model.forward_with_activations(x, t, y)
            else:
                x_pred = model(x, t, y)
                activations = None
            
            # Compute main loss
            main_loss = loss_fn(x_pred, x)
            
            # Compute dispersive loss if available
            disp_loss = 0.0
            if dispersive_loss is not None and activations is not None:
                for act in activations:
                    disp_loss += dispersive_loss(act)
                disp_loss /= len(activations)
            
            # Total loss
            total_loss = main_loss
            if dispersive_loss is not None:
                total_loss += config.training.get('dispersive_weight', 0.25) * disp_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_val)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % config.training.log_every_n_steps == 0:
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Main: {main_loss.item():.4f}, "
                    f"Disp: {disp_loss:.4f}"
                )
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "total_loss": total_loss.item(),
                        "main_loss": main_loss.item(),
                        "dispersive_loss": disp_loss,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    })
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Log epoch results
        avg_loss = epoch_loss / num_batches
        logging.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Log epoch to wandb
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "avg_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Save checkpoint
        if epoch % config.training.save_every_n_epochs == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Log checkpoint to wandb
            if wandb.run is not None:
                wandb.save(str(checkpoint_path))


def sample_from_model(model: nn.Module, config: DictConfig) -> None:
    """Sample from the trained model."""
    device = next(model.parameters()).device
    model.eval()
    
    # Create sampling configuration
    sampling_config = config.sampling
    
    with torch.no_grad():
        if sampling_config.method == "ode":
            # Import sampling function dynamically
            from dflow.sampling import sample_ode
            samples = sample_ode(
                model=model,
                shape=(config.dataset.batch_size, 3, 256, 256),
                num_steps=sampling_config.num_steps,
                device=device
            )
        elif sampling_config.method == "sde":
            # Import sampling function dynamically
            from dflow.sampling import sample_sde
            samples = sample_sde(
                model=model,
                shape=(config.dataset.batch_size, 3, 256, 256),
                num_steps=sampling_config.num_steps,
                device=device
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_config.method}")
    
    # Save samples
    output_dir = Path(config.output_dir) / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert samples to images and save
    torch.save(samples, output_dir / "samples.pt")
    logging.info(f"Samples saved to {output_dir}")
    
    # Log samples to wandb
    if wandb.run is not None:
        wandb.save(str(output_dir / "samples.pt"))


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """Main function."""
    sleep(random.randint(1, 10))
    logging.info("---------------------------------------------------------------")
    
    # Log environment
    envs = {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "PYTHONOPTIMIZE"]}
    logging.info("Env:\n%s", yaml.dump(envs))

    # Log overrides
    hydra_config = HydraConfig.get()
    logging.info("Command line args:\n%s", "\n".join(hydra_config.overrides.task))

    # Setup dir
    OmegaConf.set_struct(cfg, False)
    out_dir = Path(hydra_config.runtime.output_dir).absolute()
    logging.info("Hydra and wandb output path: %s", out_dir)
    if not cfg.get("output_dir"):
        cfg.output_dir = str(out_dir)
    logging.info("output path: %s", cfg.output_dir)

    # Setup wandb
    model_name = str(cfg.model.get("_target_", "unknown")).split(".")[-1]
    tags = [cfg.experiment.job_type, model_name]
    if "wandb" not in cfg:
        cfg.wandb = OmegaConf.create()
    if not cfg.wandb.get("tags"):
        cfg.wandb.tags = tags
    if not cfg.wandb.get("group"):
        cfg.wandb.group = cfg.project_name

    if not cfg.wandb.get("id"):
        # create id based on log directory for automatic resuming
        sha = hashlib.sha256()
        sha.update(str(out_dir).encode())
        cfg.wandb.id = sha.hexdigest()

    if not cfg.wandb.get("name"):
        cfg.wandb.name = (
            f"{cfg.project_name}_{cfg.experiment.job_type}_sd{cfg.seed:03d}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Create save directory
    cfg.save_dir = f"{cfg.checkpoint_dir}/{cfg.project_name}/{cfg.experiment.job_type}/"
    i = 1
    while os.path.exists(cfg.save_dir) and not cfg.get("override", False):
        cfg.save_dir += f"{i}/"
        i += 1

    OmegaConf.set_struct(cfg, True)
    wandb.init(
        dir=out_dir,
        **cfg.wandb,
    )

    # Resume old wandb run
    if wandb.run is not None and wandb.run.resumed:
        logging.info("Resume wandb run %s", wandb.run.path)

    # Log config and overrides
    logging.info("---------------------------------------------------------------")
    logging.info("Run config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logging.info("---------------------------------------------------------------")

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_config["hydra"] = OmegaConf.to_container(hydra_config, resolve=True)

    for k in [
        "help",
        "hydra_help",
        "hydra_logging",
        "job_logging",
        "searchpath",
        "callbacks",
        "sweeper",
    ]:
        wandb_config["hydra"].pop(k, None)
    wandb.config.update(wandb_config, allow_val_change=True)

    # Run experiment
    logging.info("---------------------------------------------------------------")

    try:
        OmegaConf.resolve(cfg)
        
        # Setup environment
        setup_environment(cfg)
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        # Create model
        model = create_model(cfg)
        model = model.to(device)
        logging.info(f"Model created: {cfg.model._target_}")
        
        # Create dataloader
        dataloader = create_dataloaders(cfg)
        logging.info(f"Dataloader created with {len(dataloader)} batches")
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(model, cfg)
        logging.info("Optimizer and scheduler created")
        
        # Run experiment based on job type
        if cfg.experiment.job_type == "train":
            logging.info("Starting training...")
            train_model(model, dataloader, optimizer, scheduler, cfg)
            logging.info("Training completed!")
            
        elif cfg.experiment.job_type == "sample":
            logging.info("Starting sampling...")
            sample_from_model(model, cfg)
            logging.info("Sampling completed!")
            
        else:
            logging.warning(f"Unknown job type: {cfg.experiment.job_type}")

        wandb.run.summary["error"] = None
        logging.info("Completed ✅")
        wandb.finish()

    except Exception as e:
        # Log error
        error_log_path = "logs/error_log.csv"
        fieldnames = ["timestamp", "job_type", "model", "error"]

        # Gather info for logging
        job_type = getattr(cfg.experiment, "job_type", str(cfg.experiment))
        model_name = getattr(cfg.model, "_target_", str(cfg.model))
        error_str = str(e)
        timestamp = datetime.now().isoformat()

        # Write to CSV (append, create header if file does not exist)
        try:
            write_header = False
            try:
                with open(error_log_path, "r", newline="") as f:
                    pass
            except FileNotFoundError:
                write_header = True

            with open(error_log_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": timestamp,
                    "job_type": job_type,
                    "model": model_name,
                    "error": error_str
                })
        except Exception as log_exc:
            logging.error(f"Failed to log error to {error_log_path}: {log_exc}")

        logging.critical(e, exc_info=True)
        if wandb.run is not None:
            wandb.run.summary["error"] = str(e)
            wandb.finish(exit_code=1)


def sync_wandb(wandb_dir: Path | str):
    """Sync wandb runs."""
    run_dirs = [f for f in Path(wandb_dir).iterdir() if "run-" in f.name]
    for run_dir in sorted(run_dirs, key=os.path.getmtime):
        logging.info("Syncing %s.", run_dir)
        subprocess.run(
            ["wandb", "sync", "--no-include-synced", "--mark-synced", str(run_dir)]
        )


if __name__ == "__main__":
    main()
