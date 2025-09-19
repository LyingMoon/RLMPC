"""
MPC module for RLMPC package.

Contains Model Predictive Control implementations including SMPC and NNMPC.
"""

from .neural_mpc import NeuralMPCTrainer, train_neural_mpc, train_neural_mpc_from_arrays
from .python_mpc import SystemModel, QubeSystemModel, CustomSystemModel, ConfigurableMPC, MPCConfig, prediction_shrinkage

__all__ = [
    'NeuralMPCTrainer',
    'train_neural_mpc',
    'train_neural_mpc_from_arrays',
    'SystemModel',
    'QubeSystemModel',
    'CustomSystemModel',
    'ConfigurableMPC',
    'MPCConfig',
    'prediction_shrinkage'
]