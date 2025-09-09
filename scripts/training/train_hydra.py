"""
Hydra-compatible training script for dFlow with modular configuration.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import wandb
from pathlib import Path

from dflow.models import SiT
from dflow.training import DispersiveLoss, CombinedLoss


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
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
    
    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Create model
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=cfg.dataset.drop_last
    )
    
    # Create optimizer
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())
    
    # Create scheduler
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    
    # Create loss function
    loss_fn = hydra.utils.instantiate(cfg.loss)
    
    # Training loop
    model.train()
    step = 0
    
    for epoch in range(cfg.training.num_epochs):
        epoch_loss = 0.0
        epoch_main_loss = 0.0
        epoch_disp_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                x = batch
                y = None
            
            x = x.to(device)
            if y is not None:
                y = y.to(device)
            
            # Sample timesteps
            t = torch.randint(0, 1000, (x.shape[0],), device=device)
            
            # Forward pass
            if isinstance(loss_fn, CombinedLoss):
                x_pred, activations = model(x, t, y, return_act=True)
                total_loss, main_loss, disp_loss = loss_fn(x_pred, x, activations)
            else:
                x_pred = model(x, t, y)
                total_loss = loss_fn(x_pred, x)
                main_loss = total_loss
                disp_loss = 0.0
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if cfg.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_main_loss += main_loss.item()
            epoch_disp_loss += disp_loss.item()
            num_batches += 1
            step += 1
            
            # Logging
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{cfg.training.num_epochs}, "
                      f"Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Step {step}, "
                      f"Loss: {total_loss.item():.4f}, "
                      f"Main: {main_loss.item():.4f}, "
                      f"Disp: {disp_loss.item():.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
                
                if cfg.wandb.enabled:
                    wandb.log({
                        "epoch": epoch,
                        "step": step,
                        "loss": total_loss.item(),
                        "main_loss": main_loss.item(),
                        "disp_loss": disp_loss.item(),
                        "lr": scheduler.get_last_lr()[0]
                    })
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_main_loss = epoch_main_loss / num_batches
        avg_disp_loss = epoch_disp_loss / num_batches
        
        print(f"Epoch {epoch+1}/{cfg.training.num_epochs} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Main Loss: {avg_main_loss:.4f}")
        print(f"  Avg Disp Loss: {avg_disp_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 50)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(cfg.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': cfg
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    print("Training completed!")
    
    # Save final model
    final_model_path = Path(cfg.checkpoint_dir) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg
    }, final_model_path)
    print(f"Saved final model: {final_model_path}")


if __name__ == "__main__":
    main()
