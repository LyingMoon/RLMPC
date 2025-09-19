"""Neural MPC training module."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler

from ..models.neural_networks import PolicyNet
from ..config import RLMPCConfig, DEFAULT_CONFIG
from ..utils.logging import TrainingLogger, get_timestamp


class MPCDataset(Dataset):
    """Dataset for MPC training data.

    Supports loading data from:
    - MATLAB .mat files containing input-output training pairs
    - Python numpy arrays (.npy files)
    - Direct numpy arrays

    Args:
        input_path: Path to input data file or numpy array
        output_path: Path to output data file or numpy array  
        input_key: Key for input data in .mat file (ignored for .npy)
        output_key: Key for output data in .mat file (ignored for .npy)
        normalize: Whether to normalize the data
        device: Device to load data on
    """

    def __init__(
        self,
        input_path,  # Can be str path or np.ndarray
        output_path,  # Can be str path or np.ndarray
        input_key: str = "INPUT",
        output_key: str = "OUTPUT",
        normalize: bool = True,
        device: str = "cpu"
    ):
        self.device = device
        self.normalize = normalize

        # Load data based on input type
        if isinstance(input_path, np.ndarray) and isinstance(output_path, np.ndarray):
            # Direct numpy arrays
            self.input_data = torch.FloatTensor(input_path).to(self.device)
            self.output_data = torch.FloatTensor(output_path).to(self.device)
        else:
            # File paths
            self.input_path = Path(input_path)
            self.output_path = Path(output_path)
            self._load_data(input_key, output_key)

        if self.normalize:
            self._normalize_data()

    def _load_data(self, input_key: str, output_key: str) -> None:
        """Load data from files (.mat or .npy)."""
        try:
            # Check file extensions
            if self.input_path.suffix == '.npy':
                input_data = np.load(self.input_path)
                output_data = np.load(self.output_path)
            else:
                # Assume MATLAB format
                input_data = scipy.io.loadmat(self.input_path)[input_key]
                output_data = scipy.io.loadmat(self.output_path)[output_key]
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Error loading data: {e}")

        if input_data.shape[0] != output_data.shape[0]:
            raise ValueError(
                f"Input and output data size mismatch: "
                f"{input_data.shape[0]} vs {output_data.shape[0]}"
            )

        self.input_data = torch.FloatTensor(input_data).to(self.device)
        self.output_data = torch.FloatTensor(output_data).to(self.device)

    def _normalize_data(self) -> None:
        """Normalize input and output data."""
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        # Fit and transform
        input_np = self.input_data.cpu().numpy()
        output_np = self.output_data.cpu().numpy()

        input_normalized = self.input_scaler.fit_transform(input_np)
        output_normalized = self.output_scaler.fit_transform(output_np)

        self.input_data = torch.FloatTensor(input_normalized).to(self.device)
        self.output_data = torch.FloatTensor(output_normalized).to(self.device)

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_data[idx], self.output_data[idx]

    def get_data_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "num_samples": len(self),
            "input_dim": self.input_data.shape[1],
            "output_dim": self.output_data.shape[1],
            "input_mean": self.input_data.mean(dim=0),
            "input_std": self.input_data.std(dim=0),
            "output_mean": self.output_data.mean(dim=0),
            "output_std": self.output_data.std(dim=0),
        }


class NeuralMPCTrainer:
    """Trainer for Neural MPC models.

    Args:
        config: Training configuration
        logger: Optional logger for training progress
    """

    def __init__(
        self,
        config: RLMPCConfig = DEFAULT_CONFIG,
        logger: Optional[TrainingLogger] = None
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logger or TrainingLogger("NeuralMPC")

        # Initialize model, optimizer, and loss function
        self.model = PolicyNet(
            n_states=config.network.n_states,
            n_hiddens=config.network.n_hiddens,
            n_actions=config.network.n_actions,
            action_bound=config.network.action_bound,
            dropout_rate=config.network.dropout_rate
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.network.learning_rate
        )
        self.loss_function = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )

        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0

    def load_data(
        self,
        input_path: str,
        output_path: str
    ) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare training data.

        Args:
            input_path: Path to input data file
            output_path: Path to output data file

        Returns:
            Training and validation data loaders
        """
        dataset = MPCDataset(
            input_path=input_path,
            output_path=output_path,
            input_key=self.config.data.input_data_key,
            output_key=self.config.data.output_data_key,
            normalize=self.config.data.normalize_data,
            device=self.device
        )

        # Log dataset statistics
        stats = dataset.get_data_stats()
        self.logger.logger.info(f"Dataset loaded: {stats['num_samples']} samples")
        self.logger.logger.info(
            f"Input dim: {stats['input_dim']}, Output dim: {stats['output_dim']}"
        )

        # Split into train/validation
        val_size = int(len(dataset) * self.config.training.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.data.shuffle
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )

        return train_loader, val_loader

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for state, target_action in dataloader:
            state = state.to(self.device)
            target_action = target_action.to(self.device)

            self.optimizer.zero_grad()
            predicted_action = self.model(state)
            loss = self.loss_function(predicted_action, target_action)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, dataloader: DataLoader) -> float:
        """Validate model.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for state, target_action in dataloader:
                state = state.to(self.device)
                target_action = target_action.to(self.device)

                predicted_action = self.model(state)
                loss = self.loss_function(predicted_action, target_action)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def should_early_stop(self, val_loss: float) -> bool:
        """Check if training should stop early.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.config.training.early_stop_threshold:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.training.early_stop_patience

    def train(
        self,
        input_path: str,
        output_path: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the Neural MPC model.

        Args:
            input_path: Path to input training data
            output_path: Path to output training data
            save_path: Path to save trained model

        Returns:
            Training results dictionary
        """
        self.logger.logger.info("Starting Neural MPC training...")
        self.logger.logger.info(f"Device: {self.device}")

        # Load data
        train_loader, val_loader = self.load_data(input_path, output_path)

        # Training loop
        train_losses = []
        val_losses = []
        best_model_state = None

        for epoch in range(self.config.training.max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log progress
            self.logger.log_epoch(epoch + 1, train_loss, current_lr)
            self.logger.log_validation(val_loss)

            # Save best model
            if val_loss < self.best_loss:
                best_model_state = self.model.state_dict().copy()

            # Check early stopping
            if self.should_early_stop(val_loss):
                self.logger.log_early_stopping(
                    epoch + 1, self.config.training.early_stop_patience
                )
                break

        # Load best model
        if best_model_state is not None and self.config.training.save_best_only:
            self.model.load_state_dict(best_model_state)

        self.logger.log_training_complete(epoch + 1, train_losses[-1])

        # Save model
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            self.logger.logger.info(f"Model saved to {save_path}")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_loss": self.best_loss,
            "epochs": epoch + 1
        }


def train_neural_mpc(
    input_path: str,
    output_path: str,
    config: Optional[RLMPCConfig] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to train Neural MPC.

    Args:
        input_path: Path to input training data
        output_path: Path to output training data
        config: Training configuration
        save_path: Path to save trained model

    Returns:
        Training results
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Auto-detect data dimensions and update config
    temp_dataset = MPCDataset(input_path, output_path, normalize=False)
    input_dim = temp_dataset.input_data.shape[1]
    output_dim = temp_dataset.output_data.shape[1]
    
    # Update network config with actual dimensions
    config.network.n_states = input_dim
    config.network.n_actions = output_dim

    # Set up logging
    timestamp = get_timestamp()
    log_file = f"results/logs/neural_mpc_{timestamp}.log" if config.save_logs else None
    logger = TrainingLogger("NeuralMPC", log_file)
    
    logger.logger.info(f"Auto-detected dimensions: input={input_dim}, output={output_dim}")

    # Train model
    trainer = NeuralMPCTrainer(config, logger)
    return trainer.train(input_path, output_path, save_path)


def train_neural_mpc_from_arrays(
    input_data: np.ndarray,
    output_data: np.ndarray,
    config: Optional[RLMPCConfig] = None,
    save_path: Optional[str] = None,
    logger: Optional[TrainingLogger] = None
) -> Dict[str, Any]:
    """Train Neural MPC directly from numpy arrays.
    
    Args:
        input_data: Input training data [N, input_dim]
        output_data: Output training data [N, output_dim]  
        config: Training configuration
        save_path: Path to save trained model
        logger: Training logger
        
    Returns:
        Training results dictionary
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    if logger is None:
        logger = TrainingLogger("NeuralMPC_Arrays", "results/logs/neural_mpc_arrays.log")
    
    # Update config based on data dimensions
    input_dim = input_data.shape[1]
    output_dim = output_data.shape[1]
    
    # Assumes: [state_dim + reference_steps] for input
    # Default to 4 states, rest are reference steps
    if input_dim > 4:
        ref_steps = input_dim - 4
        logger.logger.info(f"Detected {ref_steps} reference steps in input data")
    
    # Update network config
    config.network.n_states = input_dim  # Total input dimension
    config.network.n_actions = output_dim
    
    if save_path is None:
        save_path = f"data/models/neural_mpc_python_{get_timestamp()}.pth"
    
    logger.logger.info(f"Training Neural MPC from arrays:")
    logger.logger.info(f"  Input shape: {input_data.shape}")
    logger.logger.info(f"  Output shape: {output_data.shape}")
    logger.logger.info(f"  Save path: {save_path}")
    
    # Train model using arrays directly
    trainer = NeuralMPCTrainer(config, logger)
    return trainer.train(input_data, output_data, save_path)


if __name__ == "__main__":
    # Example usage
    config = DEFAULT_CONFIG

    # Update paths to your actual data files
    input_path = config.get_matlab_path("001INPUT3.mat")
    output_path = config.get_matlab_path("001OUTPUT3.mat")
    save_path = config.get_model_path(f"neural_mpc_{get_timestamp()}.pth")

    # Check if data files exist
    if not input_path.exists() or not output_path.exists():
        print(f"Warning: Data files not found.")
        print(f"Expected: {input_path} and {output_path}")
        print("Please update the paths in this script or provide the data files.")
    else:
        results = train_neural_mpc(
            str(input_path),
            str(output_path),
            config=config,
            save_path=str(save_path)
        )
        print(f"Training completed. Best loss: {results['best_loss']:.6f}")