"""
Models module for RLMPC package.

Contains neural networks and hardware interfaces.
"""

from .neural_networks import PolicyNet, CriticNet

__all__ = ['PolicyNet', 'CriticNet']