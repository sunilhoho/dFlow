"""
Examples demonstrating VICReg loss usage with different input formats.

This script shows how to use VICReg loss with both list and tensor formats
for representations.
"""

import torch
import torch.nn as nn
from dflow.losses import VICRegLoss, VICRegDispersiveLoss, VICRegMultiScaleLoss


def example_list_format():
    """Example using list format for representations."""
    print("=== VICReg Loss with List Format ===")
    
    # Create VICReg loss
    vicreg_loss = VICRegLoss(
        lambda_inv=25.0,
        lambda_var=25.0,
        lambda_cov=1.0,
        aggregation='mean'
    )
    
    # Create sample data
    batch_size = 8
    hidden_dim = 128
    
    # List of representations from different layers
    # Each representation has shape [B, **dims]
    z1_list = [
        torch.randn(batch_size, hidden_dim),  # Layer 1
        torch.randn(batch_size, hidden_dim),  # Layer 2
        torch.randn(batch_size, hidden_dim),  # Layer 3
    ]
    
    z2_list = [
        torch.randn(batch_size, hidden_dim),  # Augmented view 1
        torch.randn(batch_size, hidden_dim),  # Augmented view 2
        torch.randn(batch_size, hidden_dim),  # Augmented view 3
    ]
    
    # Compute VICReg loss
    loss_output = vicreg_loss(z1_list, z2_list)
    
    print(f"Total loss: {loss_output['total_loss']:.4f}")
    print(f"Invariance loss: {loss_output['invariance_loss']:.4f}")
    print(f"Variance loss: {loss_output['variance_loss']:.4f}")
    print(f"Covariance loss: {loss_output['covariance_loss']:.4f}")
    print()


def example_tensor_format():
    """Example using tensor format for representations."""
    print("=== VICReg Loss with Tensor Format ===")
    
    # Create VICReg loss
    vicreg_loss = VICRegLoss(
        lambda_inv=25.0,
        lambda_var=25.0,
        lambda_cov=1.0,
        aggregation='mean'
    )
    
    # Create sample data
    batch_size = 8
    num_layers = 3
    hidden_dim = 128
    
    # Tensor format: [B, M, D] where M is number of representations
    z1_tensor = torch.randn(batch_size, num_layers, hidden_dim)
    z2_tensor = torch.randn(batch_size, num_layers, hidden_dim)
    
    # Compute VICReg loss
    loss_output = vicreg_loss(z1_tensor, z2_tensor)
    
    print(f"Total loss: {loss_output['total_loss']:.4f}")
    print(f"Invariance loss: {loss_output['invariance_loss']:.4f}")
    print(f"Variance loss: {loss_output['variance_loss']:.4f}")
    print(f"Covariance loss: {loss_output['covariance_loss']:.4f}")
    print()


def example_dispersive_vicreg():
    """Example using combined VICReg + Dispersive loss."""
    print("=== VICReg + Dispersive Loss ===")
    
    # Create combined loss
    combined_loss = VICRegDispersiveLoss(
        lambda_vicreg=1.0,
        lambda_disp=0.25,
        vicreg_params={
            'lambda_inv': 25.0,
            'lambda_var': 25.0,
            'lambda_cov': 1.0,
            'aggregation': 'mean'
        },
        disp_tau=1.0,
        aggregation='mean'
    )
    
    # Create sample data
    batch_size = 8
    hidden_dim = 128
    
    # List format
    z1_list = [
        torch.randn(batch_size, hidden_dim),
        torch.randn(batch_size, hidden_dim),
        torch.randn(batch_size, hidden_dim),
    ]
    
    z2_list = [
        torch.randn(batch_size, hidden_dim),
        torch.randn(batch_size, hidden_dim),
        torch.randn(batch_size, hidden_dim),
    ]
    
    # Compute combined loss
    loss_output = combined_loss(z1_list, z2_list)
    
    print(f"Total loss: {loss_output['total_loss']:.4f}")
    print(f"VICReg loss: {loss_output['vicreg_loss']:.4f}")
    print(f"Dispersive loss: {loss_output['dispersive_loss']:.4f}")
    print(f"Invariance loss: {loss_output['invariance_loss']:.4f}")
    print(f"Variance loss: {loss_output['variance_loss']:.4f}")
    print(f"Covariance loss: {loss_output['covariance_loss']:.4f}")
    print()


def example_multiscale():
    """Example using multi-scale VICReg loss."""
    print("=== Multi-scale VICReg Loss ===")
    
    # Create multi-scale VICReg loss
    multiscale_loss = VICRegMultiScaleLoss(
        lambda_inv=25.0,
        lambda_var=25.0,
        lambda_cov=1.0,
        scale_weights=[1.0, 0.8, 0.6],  # Different weights for different scales
        aggregation='mean'
    )
    
    # Create sample data for multiple scales
    batch_size = 8
    hidden_dim = 128
    
    # Representations at different scales
    representations_list = [
        # Scale 1: 3 layers
        [
            torch.randn(batch_size, hidden_dim),
            torch.randn(batch_size, hidden_dim),
            torch.randn(batch_size, hidden_dim),
        ],
        # Scale 2: 2 layers
        [
            torch.randn(batch_size, hidden_dim),
            torch.randn(batch_size, hidden_dim),
        ],
        # Scale 3: 1 layer
        [
            torch.randn(batch_size, hidden_dim),
        ]
    ]
    
    z2_list = [
        # Augmented views for each scale
        [
            torch.randn(batch_size, hidden_dim),
            torch.randn(batch_size, hidden_dim),
            torch.randn(batch_size, hidden_dim),
        ],
        [
            torch.randn(batch_size, hidden_dim),
            torch.randn(batch_size, hidden_dim),
        ],
        [
            torch.randn(batch_size, hidden_dim),
        ]
    ]
    
    # Compute multi-scale loss
    loss_output = multiscale_loss(representations_list, z2_list)
    
    print(f"Total loss: {loss_output['total_loss']:.4f}")
    print(f"Invariance loss: {loss_output['invariance_loss']:.4f}")
    print(f"Variance loss: {loss_output['variance_loss']:.4f}")
    print(f"Covariance loss: {loss_output['covariance_loss']:.4f}")
    print(f"Number of scales: {len(loss_output['scale_losses'])}")
    print()


def example_aggregation_methods():
    """Example showing different aggregation methods."""
    print("=== Different Aggregation Methods ===")
    
    # Create sample data
    batch_size = 8
    hidden_dim = 128
    
    z1_list = [
        torch.randn(batch_size, hidden_dim),
        torch.randn(batch_size, hidden_dim),
        torch.randn(batch_size, hidden_dim),
    ]
    
    z2_list = [
        torch.randn(batch_size, hidden_dim),
        torch.randn(batch_size, hidden_dim),
        torch.randn(batch_size, hidden_dim),
    ]
    
    # Test different aggregation methods
    aggregation_methods = ['mean', 'sum', 'max', 'min']
    
    for method in aggregation_methods:
        vicreg_loss = VICRegLoss(
            lambda_inv=25.0,
            lambda_var=25.0,
            lambda_cov=1.0,
            aggregation=method
        )
        
        loss_output = vicreg_loss(z1_list, z2_list)
        
        print(f"{method.upper()} aggregation:")
        print(f"  Total loss: {loss_output['total_loss']:.4f}")
        print(f"  Invariance loss: {loss_output['invariance_loss']:.4f}")
        print(f"  Variance loss: {loss_output['variance_loss']:.4f}")
        print(f"  Covariance loss: {loss_output['covariance_loss']:.4f}")
        print()


def example_training_integration():
    """Example showing how to integrate VICReg loss in training."""
    print("=== Training Integration Example ===")
    
    # Simulate training loop
    batch_size = 8
    hidden_dim = 128
    num_layers = 3
    
    # Create VICReg loss
    vicreg_loss = VICRegLoss(
        lambda_inv=25.0,
        lambda_var=25.0,
        lambda_cov=1.0,
        aggregation='mean'
    )
    
    # Simulate model forward pass
    def simulate_model_forward(x):
        """Simulate model forward pass returning activations."""
        activations = []
        for i in range(num_layers):
            # Simulate layer output
            activation = torch.randn(batch_size, hidden_dim)
            activations.append(activation)
        return activations
    
    # Training loop
    for epoch in range(3):
        epoch_losses = {
            'total_loss': 0.0,
            'invariance_loss': 0.0,
            'variance_loss': 0.0,
            'covariance_loss': 0.0
        }
        
        for batch in range(5):  # Simulate 5 batches
            # Simulate input data
            x = torch.randn(batch_size, 3, 32, 32)
            
            # Forward pass
            activations = simulate_model_forward(x)
            
            # Compute VICReg loss
            loss_output = vicreg_loss(activations)
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_output[key].item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= 5
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Total loss: {epoch_losses['total_loss']:.4f}")
        print(f"  Invariance loss: {epoch_losses['invariance_loss']:.4f}")
        print(f"  Variance loss: {epoch_losses['variance_loss']:.4f}")
        print(f"  Covariance loss: {epoch_losses['covariance_loss']:.4f}")
        print()


if __name__ == "__main__":
    print("VICReg Loss Examples")
    print("=" * 50)
    print()
    
    example_list_format()
    example_tensor_format()
    example_dispersive_vicreg()
    example_multiscale()
    example_aggregation_methods()
    example_training_integration()
    
    print("All examples completed!")
