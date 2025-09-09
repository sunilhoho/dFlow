"""
Contrastive loss functions for dFlow.

This module contains various contrastive learning loss functions
that can be used for self-supervised representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning.
    
    This loss encourages similar samples to have similar representations
    and dissimilar samples to have different representations.
    """
    
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, z1, z2, labels=None):
        """
        Compute contrastive loss.
        
        Args:
            z1: First set of representations (batch_size, hidden_dim)
            z2: Second set of representations (batch_size, hidden_dim)
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Contrastive loss value
        """
        if self.normalize:
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.T) / self.temperature
        
        if labels is not None:
            # Supervised contrastive learning
            return self._supervised_contrastive_loss(sim_matrix, labels)
        else:
            # Self-supervised contrastive learning
            return self._self_supervised_contrastive_loss(sim_matrix)
    
    def _supervised_contrastive_loss(self, sim_matrix, labels):
        """Compute supervised contrastive loss."""
        batch_size = sim_matrix.size(0)
        
        # Create positive mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Only consider positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def _self_supervised_contrastive_loss(self, sim_matrix):
        """Compute self-supervised contrastive loss."""
        batch_size = sim_matrix.size(0)
        
        # Labels are the diagonal indices (positive pairs)
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class InfoNCE(nn.Module):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss.
    
    This is a variant of contrastive learning that maximizes mutual information
    between positive pairs while minimizing it for negative pairs.
    """
    
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, z1, z2):
        """
        Compute InfoNCE loss.
        
        Args:
            z1: First set of representations (batch_size, hidden_dim)
            z2: Second set of representations (batch_size, hidden_dim)
            
        Returns:
            InfoNCE loss value
        """
        if self.normalize:
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.T) / self.temperature
        
        # Positive pairs are on the diagonal
        positive_sim = torch.diag(sim_matrix)
        
        # Compute loss
        log_prob = positive_sim - torch.log(torch.exp(sim_matrix).sum(dim=1))
        loss = -log_prob.mean()
        
        return loss


class SimCLRLoss(nn.Module):
    """
    SimCLR (Simple Contrastive Learning of Representations) loss.
    
    This is the loss function used in the SimCLR paper for self-supervised learning.
    """
    
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, z1, z2):
        """
        Compute SimCLR loss.
        
        Args:
            z1: First set of representations (batch_size, hidden_dim)
            z2: Second set of representations (batch_size, hidden_dim)
            
        Returns:
            SimCLR loss value
        """
        if self.normalize:
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature
        
        # Create positive mask (diagonal blocks)
        batch_size = z1.size(0)
        mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device)
        mask[:batch_size, batch_size:] = 1
        mask[batch_size:, :batch_size] = 1
        
        # Remove diagonal
        mask = mask - torch.eye(2 * batch_size, device=z.device)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Only consider positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss
