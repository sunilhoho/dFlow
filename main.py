#!/usr/bin/env python3
"""
Main entry point for dFlow (Dispersive Flow).
"""

import os
import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add dflow to path
sys.path.insert(0, str(Path(__file__).parent))

from dflow.models import SiT
from dflow.losses import DispersiveLoss, VICRegLoss
from dflow.utils import create_dataloader_from_config
from dflow.sampling import sample_ode, sample_sde


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_environment(config: DictConfig):
    """Setup environment and seeds."""
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(config: DictConfig) -> None:
    """Main function."""
    setup_logging()
    setup_environment(config)
    
    logging.info("Configuration:")
    logging.info(OmegaConf.to_yaml(config))
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create model
    model = hydra.utils.instantiate(config.model)
    model = model.to(device)
    logging.info(f"Model created: {config.model._target_}")
    
    # Create dataloader
    dataloader = create_dataloader_from_config(config.dataset)
    logging.info(f"Dataloader created with {len(dataloader)} batches")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=1e-2
    )
    
    # Create scheduler
    scheduler = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)
    
    # Create loss function
    loss_fn = hydra.utils.instantiate(config.loss)
    
    # Create dispersive loss
    dispersive_loss = DispersiveLoss(lambda_disp=0.25, tau=1.0)
    
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
            
            # Compute dispersive loss
            disp_loss = 0.0
            if activations is not None:
                for act in activations:
                    disp_loss += dispersive_loss(act)
                disp_loss /= len(activations)
            
            # Total loss
            total_loss = main_loss + 0.25 * disp_loss
            
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
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Log epoch results
        avg_loss = epoch_loss / num_batches
        logging.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
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


if __name__ == "__main__":
    main()
