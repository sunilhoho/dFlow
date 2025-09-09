"""
Configuration management for dFlow models.

This module provides access to model, training, and loss configurations.
"""

from ..configs.models.sit_configs import (
    SIT_CONFIGS,
    TRAINING_CONFIGS, 
    DISPERSIVE_LOSS_CONFIGS,
    get_model_config,
    get_training_config,
    get_dispersive_loss_config
)

__all__ = [
    'SIT_CONFIGS',
    'TRAINING_CONFIGS',
    'DISPERSIVE_LOSS_CONFIGS', 
    'get_model_config',
    'get_training_config',
    'get_dispersive_loss_config'
]
