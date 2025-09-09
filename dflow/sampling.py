"""
Sampling utilities for SiT models.

This module provides ODE and SDE sampling methods for generated images.
"""

import torch
import numpy as np
from .models import SiT

def sample_ode(model, shape, num_steps=100, device='cuda', cfg_scale=1.0):
    """
    Sample images using ODE (Ordinary Differential Equation) method.
    
    Args:
        model: Trained SiT model
        shape: Output shape (batch_size, channels, height, width)
        num_steps: Number of sampling steps
        device: Device to sample on
        cfg_scale: Classifier-free guidance scale
        
    Returns:
        Generated images
    """
    model.eval()
    batch_size = shape[0]
    
    # Sample random noise
    x = torch.randn(shape, device=device)
    
    # Sample class labels
    y = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Create timesteps
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    with torch.no_grad():
        for i in range(num_steps):
            t = timesteps[i:i+1].repeat(batch_size)
            
            if cfg_scale > 1.0:
                x_pred = model.forward_with_cfg(x, t, y, cfg_scale)
            else:
                x_pred = model(x, t, y)
            
            # Simple Euler step
            dt = timesteps[i+1] - timesteps[i]
            x = x + dt * x_pred
    
    return x

def sample_sde(model, shape, num_steps=100, device='cuda', cfg_scale=1.0, noise_schedule='linear'):
    """
    Sample images using SDE (Stochastic Differential Equation) method.
    
    Args:
        model: Trained SiT model
        shape: Output shape (batch_size, channels, height, width)
        num_steps: Number of sampling steps
        device: Device to sample on
        cfg_scale: Classifier-free guidance scale
        noise_schedule: Noise schedule type
        
    Returns:
        Generated images
    """
    model.eval()
    batch_size = shape[0]
    
    # Sample random noise
    x = torch.randn(shape, device=device)
    
    # Sample class labels
    y = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Create timesteps
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    with torch.no_grad():
        for i in range(num_steps):
            t = timesteps[i:i+1].repeat(batch_size)
            
            if cfg_scale > 1.0:
                x_pred = model.forward_with_cfg(x, t, y, cfg_scale)
            else:
                x_pred = model(x, t, y)
            
            # SDE step with noise
            dt = timesteps[i+1] - timesteps[i]
            noise_scale = np.sqrt(2 * dt)  # Simple noise schedule
            
            # Add noise for SDE
            noise = torch.randn_like(x) * noise_scale
            x = x + dt * x_pred + noise
    
    return x
