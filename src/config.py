"""Configuration management for RLMPC."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
from pathlib import Path


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    n_states: int = 4
    n_hiddens: int = 128
    n_actions: int = 1
    action_bound: float = 15.0
    dropout_rate: float = 0.0
    learning_rate: float = 0.001


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    max_epochs: int = 100
    early_stop_patience: int = 10
    early_stop_threshold: float = 0.2
    validation_split: float = 0.2
    save_best_only: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    input_data_key: str = "INPUT"
    output_data_key: str = "OUTPUT"
    normalize_data: bool = True
    shuffle: bool = True


@dataclass
class RLMPCConfig:
    """Main configuration class."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Paths (relative to project root)
    data_dir: str = "data"
    models_dir: str = "data/models"
    matlab_dir: str = "data/matlab"
    results_dir: str = "results"

    # Logging
    log_level: str = "INFO"
    save_logs: bool = True

    # Device
    device: str = "auto"  # auto, cpu, cuda

    def __post_init__(self):
        """Post-initialization validation."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLMPCConfig':
        """Create config from dictionary."""
        network_config = NetworkConfig(**config_dict.get('network', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))

        return cls(
            network=network_config,
            training=training_config,
            data=data_config,
            **{k: v for k, v in config_dict.items()
               if k not in ['network', 'training', 'data']}
        )

    def get_data_path(self, filename: str) -> Path:
        """Get full path to data file."""
        return Path(self.data_dir) / filename

    def get_model_path(self, filename: str) -> Path:
        """Get full path to model file."""
        return Path(self.models_dir) / filename

    def get_matlab_path(self, filename: str) -> Path:
        """Get full path to MATLAB file."""
        return Path(self.matlab_dir) / filename


# Default configuration instance
DEFAULT_CONFIG = RLMPCConfig()