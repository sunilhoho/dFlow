"""
VICReg (Variance-Invariance-Covariance Regularization) Loss implementation.

VICReg is a self-supervised learning method that encourages representations to be
invariant to augmentations while maintaining variance and decorrelating features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """
    VICReg Loss implementation for self-supervised learning.
    
    VICReg encourages:
    1. Invariance: Similar representations for augmented views
    2. Variance: High variance across the batch
    3. Covariance: Decorrelated features
    
    Supports both:
    - List of representations: [rep1, rep2, ...] where each rep has shape [B, **dims]
    - Tensor of representations: [B, M, **dims] where M is number of representations
    """
    
    def __init__(self, 
                 lambda_inv=25.0, 
                 lambda_var=25.0, 
                 lambda_cov=1.0,
                 sim_coeff=25.0,
                 std_coeff=25.0,
                 cov_coeff=1.0,
                 eps=1e-4,
                 aggregation='mean'):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps
        self.aggregation = aggregation  # 'mean', 'sum', 'max', 'min'
    
    def forward(self, z1, z2=None):
        """
        Compute VICReg loss.
        
        Args:
            z1: First set of representations. Can be:
                - List of tensors: [rep1, rep2, ...] where each has shape [B, **dims]
                - Tensor: [B, M, **dims] where M is number of representations
            z2: Second set of representations (optional) for invariance loss.
                Same format as z1.
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # Process z1
        z1_processed = self._process_representations(z1)
        
        # Process z2 if provided
        z2_processed = None
        if z2 is not None:
            z2_processed = self._process_representations(z2)
        
        # Compute losses
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        
        if z2_processed is not None:
            # Invariance loss between z1 and z2
            inv_loss = self._compute_invariance_loss(z1_processed, z2_processed)
            
            # Variance and covariance losses for both z1 and z2
            var_loss = (self._compute_variance_loss(z1_processed) + 
                       self._compute_variance_loss(z2_processed)) / 2
            cov_loss = (self._compute_covariance_loss(z1_processed) + 
                       self._compute_covariance_loss(z2_processed)) / 2
        else:
            # Only variance and covariance losses for z1
            var_loss = self._compute_variance_loss(z1_processed)
            cov_loss = self._compute_covariance_loss(z1_processed)
        
        # Total loss
        total_loss = (self.sim_coeff * inv_loss + 
                     self.std_coeff * var_loss + 
                     self.cov_coeff * cov_loss)
        
        return {
            'total_loss': total_loss,
            'invariance_loss': inv_loss,
            'variance_loss': var_loss,
            'covariance_loss': cov_loss
        }
    
    def _process_representations(self, z):
        """
        Process representations to handle both list and tensor formats.
        
        Args:
            z: Representations as list or tensor
            
        Returns:
            Processed tensor with shape [B, M, D] where:
            - B is batch size
            - M is number of representations
            - D is feature dimension
        """
        if isinstance(z, (list, tuple)):
            # List of representations: [rep1, rep2, ...]
            # Each rep has shape [B, **dims]
            processed_reps = []
            for rep in z:
                # Flatten spatial dimensions if needed
                if rep.dim() > 2:
                    rep_flat = rep.reshape(rep.size(0), -1)
                else:
                    rep_flat = rep
                processed_reps.append(rep_flat)
            
            # Stack along new dimension: [B, M, D]
            z_processed = torch.stack(processed_reps, dim=1)
            
        elif isinstance(z, torch.Tensor):
            if z.dim() == 2:
                # Single representation: [B, D] -> [B, 1, D]
                z_processed = z.unsqueeze(1)
            elif z.dim() == 3:
                # Multiple representations: [B, M, **dims]
                # Flatten feature dimensions
                B, M = z.size(0), z.size(1)
                z_flat = z.reshape(B, M, -1)
                z_processed = z_flat
            else:
                raise ValueError(f"Unsupported tensor dimension: {z.dim()}. "
                               f"Expected 2D [B, D] or 3D [B, M, **dims]")
        else:
            raise ValueError(f"Unsupported input type: {type(z)}. "
                           f"Expected list, tuple, or torch.Tensor")

        return z_processed

    def _compute_invariance_loss(self, z1, z2):
        """Compute invariance loss between two sets of representations."""
        # z1 and z2 have shape [B, M, D]
        B, M, D = z1.shape

        # Compute MSE loss for each representation pair
        losses = []
        for i in range(M):
            loss = F.mse_loss(z1[:, i], z2[:, i])
            losses.append(loss)

        # Aggregate losses across representations
        if self.aggregation == 'mean':
            return torch.stack(losses).mean()
        elif self.aggregation == 'sum':
            return torch.stack(losses).sum()
        elif self.aggregation == 'max':
            return torch.stack(losses).max()
        elif self.aggregation == 'min':
            return torch.stack(losses).min()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _compute_variance_loss(self, z):
        """Compute variance loss to encourage high variance across batch."""
        # z has shape [B, M, D]
        B, M, D = z.shape
        
        # Compute variance across batch dimension for each representation
        losses = []
        for i in range(M):
            rep = z[:, i]  # [B, D]
            var = torch.var(rep, dim=0)  # [D]
            # Hinge loss: penalize when variance is below threshold
            var_loss = F.relu(1.0 - torch.sqrt(var + self.eps)).mean()
            losses.append(var_loss)
        
        # Aggregate losses across representations
        if self.aggregation == 'mean':
            return torch.stack(losses).mean()
        elif self.aggregation == 'sum':
            return torch.stack(losses).sum()
        elif self.aggregation == 'max':
            return torch.stack(losses).max()
        elif self.aggregation == 'min':
            return torch.stack(losses).min()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _compute_covariance_loss(self, z):
        """Compute covariance loss to decorrelate features."""
        # z has shape [B, M, D]
        B, M, D = z.shape
        
        # Compute covariance loss for each representation
        losses = []
        for i in range(M):
            rep = z[:, i]  # [B, D]
            # Center the representations
            rep_centered = rep - rep.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov = torch.mm(rep_centered.T, rep_centered) / (B - 1)
            
            # Off-diagonal elements should be close to zero
            cov_loss = self._off_diagonal(cov).pow(2).sum() / D
            losses.append(cov_loss)
        
        # Aggregate losses across representations
        if self.aggregation == 'mean':
            return torch.stack(losses).mean()
        elif self.aggregation == 'sum':
            return torch.stack(losses).sum()
        elif self.aggregation == 'max':
            return torch.stack(losses).max()
        elif self.aggregation == 'min':
            return torch.stack(losses).min()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _off_diagonal(self, x):
        """Return off-diagonal elements of a square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegDispersiveLoss(nn.Module):
    """
    Combined VICReg and Dispersive Loss for diffusion models.
    
    This combines the benefits of VICReg (variance, invariance, covariance)
    with dispersive loss for better representation learning.
    
    Supports both list and tensor formats for representations.
    """

    def __init__(self, 
                 lambda_vicreg=1.0,
                 lambda_disp=0.25,
                 vicreg_params=None,
                 disp_tau=1.0,
                 aggregation='mean'):
        super().__init__()
        self.lambda_vicreg = lambda_vicreg
        self.lambda_disp = lambda_disp
        
        # VICReg parameters
        vicreg_params = vicreg_params or {}
        vicreg_params['aggregation'] = aggregation
        self.vicreg_loss = VICRegLoss(**vicreg_params)
        
        # Dispersive loss parameters
        self.disp_tau = disp_tau
        self.aggregation = aggregation
    
    def forward(self, z1, z2=None):
        """
        Compute combined VICReg and Dispersive loss.
        
        Args:
            z1: First set of representations (list or tensor)
            z2: Second set of representations (optional)
            
        Returns:
            Dictionary containing all loss components
        """
        # VICReg loss
        vicreg_output = self.vicreg_loss(z1, z2)
        
        # Dispersive loss
        disp_loss = self._compute_dispersive_loss(z1)
        if z2 is not None:
            disp_loss += self._compute_dispersive_loss(z2)
            disp_loss = disp_loss / 2
        
        # Combined loss
        total_loss = (self.lambda_vicreg * vicreg_output['total_loss'] + 
                     self.lambda_disp * disp_loss)
        
        return {
            'total_loss': total_loss,
            'vicreg_loss': vicreg_output['total_loss'],
            'dispersive_loss': disp_loss,
            'invariance_loss': vicreg_output['invariance_loss'],
            'variance_loss': vicreg_output['variance_loss'],
            'covariance_loss': vicreg_output['covariance_loss']
        }
    
    def _compute_dispersive_loss(self, z):
        """Compute dispersive loss component."""
        # Process representations
        z_processed = self._process_representations(z)
        B, M, D = z_processed.shape
        
        # Compute dispersive loss for each representation
        losses = []
        for i in range(M):
            rep = z_processed[:, i]  # [B, D]
            # Compute pairwise distances
            diff = F.pdist(rep).pow(2) / D
            # Apply temperature scaling
            diff = diff / self.disp_tau
            # Calculate loss
            loss = torch.log(torch.exp(-diff).mean())
            losses.append(loss)
        
        # Aggregate losses across representations
        if self.aggregation == 'mean':
            return torch.stack(losses).mean()
        elif self.aggregation == 'sum':
            return torch.stack(losses).sum()
        elif self.aggregation == 'max':
            return torch.stack(losses).max()
        elif self.aggregation == 'min':
            return torch.stack(losses).min()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _process_representations(self, z):
        """Process representations to handle both list and tensor formats."""
        if isinstance(z, (list, tuple)):
            # List of representations
            processed_reps = []
            for rep in z:
                if rep.dim() > 2:
                    rep_flat = rep.reshape(rep.size(0), -1)
                else:
                    rep_flat = rep
                processed_reps.append(rep_flat)
            z_processed = torch.stack(processed_reps, dim=1)
        elif isinstance(z, torch.Tensor):
            if z.dim() == 2:
                z_processed = z.unsqueeze(1)
            elif z.dim() == 3:
                B, M = z.size(0), z.size(1)
                z_flat = z.reshape(B, M, -1)
                z_processed = z_flat
            else:
                raise ValueError(f"Unsupported tensor dimension: {z.dim()}")
        else:
            raise ValueError(f"Unsupported input type: {type(z)}")
        
        return z_processed


class VICRegMultiScaleLoss(nn.Module):
    """
    Multi-scale VICReg loss that operates on representations at different scales.
    
    This is useful for diffusion models where representations are computed
    at multiple layers or scales.
    """
    
    def __init__(self, 
                 lambda_inv=25.0,
                 lambda_var=25.0,
                 lambda_cov=1.0,
                 scale_weights=None,
                 aggregation='mean'):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.scale_weights = scale_weights
        self.aggregation = aggregation
        
        # Create VICReg loss
        self.vicreg_loss = VICRegLoss(
            lambda_inv=lambda_inv,
            lambda_var=lambda_var,
            lambda_cov=lambda_cov,
            aggregation=aggregation
        )
    
    def forward(self, representations_list, z2_list=None):
        """
        Compute multi-scale VICReg loss.
        
        Args:
            representations_list: List of representation lists/tensors at different scales
            z2_list: Optional second set of representations at different scales
            
        Returns:
            Dictionary containing loss components
        """
        if z2_list is not None:
            assert len(representations_list) == len(z2_list), \
                "Number of scales must match between z1 and z2"
        
        # Compute VICReg loss at each scale
        scale_losses = []
        for i, (z1_scale, z2_scale) in enumerate(zip(representations_list, z2_list or [None] * len(representations_list))):
            vicreg_output = self.vicreg_loss(z1_scale, z2_scale)
            scale_losses.append(vicreg_output)
        
        # Aggregate losses across scales
        if self.scale_weights is None:
            scale_weights = [1.0] * len(scale_losses)
        else:
            scale_weights = self.scale_weights
        
        # Weighted aggregation
        total_inv_loss = sum(w * loss['invariance_loss'] for w, loss in zip(scale_weights, scale_losses))
        total_var_loss = sum(w * loss['variance_loss'] for w, loss in zip(scale_weights, scale_losses))
        total_cov_loss = sum(w * loss['covariance_loss'] for w, loss in zip(scale_weights, scale_losses))
        
        total_loss = (self.lambda_inv * total_inv_loss + 
                     self.lambda_var * total_var_loss + 
                     self.lambda_cov * total_cov_loss)
        
        return {
            'total_loss': total_loss,
            'invariance_loss': total_inv_loss,
            'variance_loss': total_var_loss,
            'covariance_loss': total_cov_loss,
            'scale_losses': scale_losses
        }
