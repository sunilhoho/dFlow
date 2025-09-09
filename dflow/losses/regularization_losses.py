"""
Regularization loss functions for dFlow.

This module contains various regularization loss functions
that can be used to improve model training and generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Regularization(nn.Module):
    """
    L2 regularization loss.
    
    This loss penalizes large weights to prevent overfitting.
    """
    
    def __init__(self, weight_decay=1e-4):
        super().__init__()
        self.weight_decay = weight_decay
    
    def forward(self, model):
        """
        Compute L2 regularization loss.
        
        Args:
            model: PyTorch model
            
        Returns:
            L2 regularization loss value
        """
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        return self.weight_decay * l2_reg


class L1Regularization(nn.Module):
    """
    L1 regularization loss.
    
    This loss encourages sparsity in the model weights.
    """
    
    def __init__(self, weight_decay=1e-4):
        super().__init__()
        self.weight_decay = weight_decay
    
    def forward(self, model):
        """
        Compute L1 regularization loss.
        
        Args:
            model: PyTorch model
            
        Returns:
            L1 regularization loss value
        """
        l1_reg = 0.0
        for param in model.parameters():
            l1_reg += torch.norm(param, p=1)
        return self.weight_decay * l1_reg


class SpectralRegularization(nn.Module):
    """
    Spectral regularization loss.
    
    This loss penalizes large singular values to improve generalization.
    """
    
    def __init__(self, weight_decay=1e-4):
        super().__init__()
        self.weight_decay = weight_decay
    
    def forward(self, model):
        """
        Compute spectral regularization loss.
        
        Args:
            model: PyTorch model
            
        Returns:
            Spectral regularization loss value
        """
        spectral_reg = 0.0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Compute spectral norm
                weight = module.weight
                u, s, v = torch.svd(weight)
                spectral_reg += torch.max(s)
        return self.weight_decay * spectral_reg


class GradientPenalty(nn.Module):
    """
    Gradient penalty for training stability.
    
    This loss penalizes large gradients to improve training stability.
    """
    
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, x, x_pred):
        """
        Compute gradient penalty.
        
        Args:
            x: Input tensor
            x_pred: Predicted tensor
            
        Returns:
            Gradient penalty loss value
        """
        # Compute gradients
        x.requires_grad_(True)
        grad = torch.autograd.grad(
            outputs=x_pred,
            inputs=x,
            grad_outputs=torch.ones_like(x_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        grad_norm = torch.norm(grad, p=2, dim=1)
        penalty = torch.mean((grad_norm - 1) ** 2)
        
        return self.lambda_gp * penalty


class ConsistencyRegularization(nn.Module):
    """
    Consistency regularization loss.
    
    This loss encourages consistent predictions for augmented inputs.
    """
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred1, pred2):
        """
        Compute consistency regularization loss.
        
        Args:
            pred1: First prediction
            pred2: Second prediction (augmented)
            
        Returns:
            Consistency regularization loss value
        """
        return self.weight * F.mse_loss(pred1, pred2)


class EntropyRegularization(nn.Module):
    """
    Entropy regularization loss.
    
    This loss encourages diversity in the model's predictions.
    """
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits):
        """
        Compute entropy regularization loss.
        
        Args:
            logits: Model logits
            
        Returns:
            Entropy regularization loss value
        """
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return -self.weight * entropy.mean()


class KLDivergenceRegularization(nn.Module):
    """
    KL divergence regularization loss.
    
    This loss encourages the model's output distribution to be close to a target distribution.
    """
    
    def __init__(self, target_dist=None, weight=1.0):
        super().__init__()
        self.target_dist = target_dist
        self.weight = weight
    
    def forward(self, pred_dist):
        """
        Compute KL divergence regularization loss.
        
        Args:
            pred_dist: Predicted distribution (logits)
            
        Returns:
            KL divergence regularization loss value
        """
        if self.target_dist is None:
            # Use uniform distribution as target
            target_dist = torch.ones_like(pred_dist) / pred_dist.size(1)
        else:
            target_dist = self.target_dist
        
        # Compute KL divergence
        kl_div = F.kl_div(
            F.log_softmax(pred_dist, dim=1),
            target_dist,
            reduction='batchmean'
        )
        
        return self.weight * kl_div
