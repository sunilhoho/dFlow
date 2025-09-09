"""
Dispersive Loss implementation for encouraging representation dispersion.

This loss encourages internal representations to disperse in the hidden space,
analogous to contrastive self-supervised learning, but without requiring positive pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DispersiveLoss(nn.Module):
    """
    Dispersive Loss implementation for encouraging representation dispersion.
    
    This loss encourages internal representations to disperse in the hidden space,
    analogous to contrastive self-supervised learning, but without requiring positive pairs.
    """
    
    def __init__(self, lambda_disp=0.25, tau=1.0):
        super().__init__()
        self.lambda_disp = lambda_disp
        self.tau = tau
    
    def forward(self, z):
        """
        Compute dispersive loss.
        
        Args:
            z: Hidden representations of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Loss value
        """
        # Flatten spatial dimensions
        z = z.reshape((z.shape[0], -1))
        
        # Compute pairwise distances
        diff = F.pdist(z).pow(2) / z.shape[1]
        
        # Match JAX implementation of full BxB matrix
        diff = torch.cat((diff, diff, torch.zeros(z.shape[0], device=z.device)))
        
        # Calculate loss
        loss = torch.log(torch.exp(-diff / self.tau).mean())
        
        return self.lambda_disp * loss


class DispersiveLossV2(nn.Module):
    """
    Alternative implementation of Dispersive Loss with different normalization.
    """
    
    def __init__(self, lambda_disp=0.25, tau=1.0, normalize=True):
        super().__init__()
        self.lambda_disp = lambda_disp
        self.tau = tau
        self.normalize = normalize
    
    def forward(self, z):
        """
        Compute dispersive loss with optional normalization.
        
        Args:
            z: Hidden representations of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Loss value
        """
        # Flatten spatial dimensions
        z = z.reshape((z.shape[0], -1))
        
        if self.normalize:
            z = F.normalize(z, p=2, dim=1)
        
        # Compute pairwise distances
        diff = F.pdist(z).pow(2)
        
        # Apply temperature scaling
        diff = diff / self.tau
        
        # Calculate loss
        loss = torch.log(torch.exp(-diff).mean())
        
        return self.lambda_disp * loss
