"""
Combined loss functions for dFlow.

This module contains loss functions that combine multiple loss components
for more effective training of diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dispersive_loss import DispersiveLoss
from .vicreg_loss import VICRegLoss, VICRegDispersiveLoss


class CombinedLoss(nn.Module):
    """
    Combined loss that combines a main loss with dispersive loss.
    
    This allows for flexible weighting of different loss components.
    """
    
    def __init__(self, main_loss, dispersive_loss, weight_main=1.0, weight_dispersive=1.0):
        super().__init__()
        self.main_loss = main_loss
        self.dispersive_loss = dispersive_loss
        self.weight_main = weight_main
        self.weight_dispersive = weight_dispersive
    
    def forward(self, x_pred, x_true, activations=None):
        """
        Compute combined loss.
        
        Args:
            x_pred: Predicted outputs
            x_true: True outputs
            activations: List of hidden activations for dispersive loss
            
        Returns:
            Combined loss value
        """
        # Main loss
        main_loss_value = self.main_loss(x_pred, x_true)
        
        # Dispersive loss
        disp_loss_value = 0.0
        if activations is not None and self.dispersive_loss is not None:
            for act in activations:
                disp_loss_value += self.dispersive_loss(act)
            disp_loss_value /= len(activations)
        
        # Combined loss
        total_loss = (self.weight_main * main_loss_value + 
                     self.weight_dispersive * disp_loss_value)
        
        return total_loss, main_loss_value, disp_loss_value


class VICRegCombinedLoss(nn.Module):
    """
    Combined loss that combines a main loss with VICReg loss.
    
    Supports both list and tensor formats for representations.
    """
    
    def __init__(self, main_loss, vicreg_loss, weight_main=1.0, weight_vicreg=1.0):
        super().__init__()
        self.main_loss = main_loss
        self.vicreg_loss = vicreg_loss
        self.weight_main = weight_main
        self.weight_vicreg = weight_vicreg
    
    def forward(self, x_pred, x_true, z1, z2=None):
        """
        Compute combined loss with VICReg.
        
        Args:
            x_pred: Predicted outputs
            x_true: True outputs
            z1: First set of representations (list or tensor)
            z2: Second set of representations (optional)
            
        Returns:
            Dictionary containing all loss components
        """
        # Main loss
        main_loss_value = self.main_loss(x_pred, x_true)
        
        # VICReg loss
        vicreg_output = self.vicreg_loss(z1, z2)
        
        # Combined loss
        total_loss = (self.weight_main * main_loss_value + 
                     self.weight_vicreg * vicreg_output['total_loss'])
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss_value,
            'vicreg_loss': vicreg_output['total_loss'],
            'invariance_loss': vicreg_output['invariance_loss'],
            'variance_loss': vicreg_output['variance_loss'],
            'covariance_loss': vicreg_output['covariance_loss']
        }


class MultiComponentLoss(nn.Module):
    """
    Multi-component loss that combines multiple loss functions.
    
    This allows for flexible combination of main loss, dispersive loss,
    VICReg loss, and other regularization terms.
    
    Supports both list and tensor formats for representations.
    """
    
    def __init__(self, 
                 main_loss,
                 dispersive_loss=None,
                 vicreg_loss=None,
                 regularization_loss=None,
                 weights=None):
        super().__init__()
        self.main_loss = main_loss
        self.dispersive_loss = dispersive_loss
        self.vicreg_loss = vicreg_loss
        self.regularization_loss = regularization_loss
        
        # Default weights
        weights = weights or {}
        self.weight_main = weights.get('main', 1.0)
        self.weight_disp = weights.get('dispersive', 0.0)
        self.weight_vicreg = weights.get('vicreg', 0.0)
        self.weight_reg = weights.get('regularization', 0.0)
    
    def forward(self, x_pred, x_true, activations=None, z1=None, z2=None):
        """
        Compute multi-component loss.
        
        Args:
            x_pred: Predicted outputs
            x_true: True outputs
            activations: List of hidden activations for dispersive loss
            z1: First set of representations for VICReg (list or tensor)
            z2: Second set of representations for VICReg (optional)
            
        Returns:
            Dictionary containing all loss components
        """
        losses = {}
        
        # Main loss
        main_loss_value = self.main_loss(x_pred, x_true)
        losses['main_loss'] = main_loss_value
        total_loss = self.weight_main * main_loss_value
        
        # Dispersive loss
        if self.dispersive_loss is not None and activations is not None:
            disp_loss_value = 0.0
            for act in activations:
                disp_loss_value += self.dispersive_loss(act)
            disp_loss_value /= len(activations)
            losses['dispersive_loss'] = disp_loss_value
            total_loss += self.weight_disp * disp_loss_value
        else:
            losses['dispersive_loss'] = 0.0
        
        # VICReg loss
        if self.vicreg_loss is not None and z1 is not None:
            vicreg_output = self.vicreg_loss(z1, z2)
            losses['vicreg_loss'] = vicreg_output['total_loss']
            losses['invariance_loss'] = vicreg_output['invariance_loss']
            losses['variance_loss'] = vicreg_output['variance_loss']
            losses['covariance_loss'] = vicreg_output['covariance_loss']
            total_loss += self.weight_vicreg * vicreg_output['total_loss']
        else:
            losses['vicreg_loss'] = 0.0
            losses['invariance_loss'] = 0.0
            losses['variance_loss'] = 0.0
            losses['covariance_loss'] = 0.0
        
        # Regularization loss
        if self.regularization_loss is not None:
            reg_loss_value = self.regularization_loss(x_pred)
            losses['regularization_loss'] = reg_loss_value
            total_loss += self.weight_reg * reg_loss_value
        else:
            losses['regularization_loss'] = 0.0
        
        losses['total_loss'] = total_loss
        return losses


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss that operates on representations at different scales.
    
    This is particularly useful for diffusion models where representations
    are computed at multiple layers or scales.
    """
    
    def __init__(self, 
                 main_loss,
                 dispersive_loss=None,
                 vicreg_loss=None,
                 scale_weights=None,
                 weights=None):
        super().__init__()
        self.main_loss = main_loss
        self.dispersive_loss = dispersive_loss
        self.vicreg_loss = vicreg_loss
        self.scale_weights = scale_weights
        
        # Default weights
        weights = weights or {}
        self.weight_main = weights.get('main', 1.0)
        self.weight_disp = weights.get('dispersive', 0.0)
        self.weight_vicreg = weights.get('vicreg', 0.0)
    
    def forward(self, x_pred, x_true, activations_list=None, z1_list=None, z2_list=None):
        """
        Compute multi-scale loss.
        
        Args:
            x_pred: Predicted outputs
            x_true: True outputs
            activations_list: List of activation lists at different scales
            z1_list: List of representation lists at different scales
            z2_list: List of second representation lists at different scales (optional)
            
        Returns:
            Dictionary containing all loss components
        """
        losses = {}
        
        # Main loss
        main_loss_value = self.main_loss(x_pred, x_true)
        losses['main_loss'] = main_loss_value
        total_loss = self.weight_main * main_loss_value
        
        # Dispersive loss across scales
        if self.dispersive_loss is not None and activations_list is not None:
            disp_loss_value = 0.0
            scale_weights = self.scale_weights or [1.0] * len(activations_list)
            
            for i, (activations, weight) in enumerate(zip(activations_list, scale_weights)):
                scale_disp_loss = 0.0
                for act in activations:
                    scale_disp_loss += self.dispersive_loss(act)
                scale_disp_loss /= len(activations)
                disp_loss_value += weight * scale_disp_loss
            
            disp_loss_value /= len(activations_list)
            losses['dispersive_loss'] = disp_loss_value
            total_loss += self.weight_disp * disp_loss_value
        else:
            losses['dispersive_loss'] = 0.0
        
        # VICReg loss across scales
        if self.vicreg_loss is not None and z1_list is not None:
            vicreg_loss_value = 0.0
            scale_weights = self.scale_weights or [1.0] * len(z1_list)
            
            for i, (z1_scale, z2_scale, weight) in enumerate(zip(z1_list, z2_list or [None] * len(z1_list), scale_weights)):
                vicreg_output = self.vicreg_loss(z1_scale, z2_scale)
                vicreg_loss_value += weight * vicreg_output['total_loss']
            
            vicreg_loss_value /= len(z1_list)
            losses['vicreg_loss'] = vicreg_loss_value
            losses['invariance_loss'] = vicreg_output['invariance_loss']
            losses['variance_loss'] = vicreg_output['variance_loss']
            losses['covariance_loss'] = vicreg_output['covariance_loss']
            total_loss += self.weight_vicreg * vicreg_loss_value
        else:
            losses['vicreg_loss'] = 0.0
            losses['invariance_loss'] = 0.0
            losses['variance_loss'] = 0.0
            losses['covariance_loss'] = 0.0
        
        losses['total_loss'] = total_loss
        return losses
