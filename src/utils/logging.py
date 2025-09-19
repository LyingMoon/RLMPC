"""Logging utilities for RLMPC."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class TrainingLogger:
    """Logger for training progress."""

    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = setup_logger(name, log_file=log_file)
        self.epoch_losses = []
        self.best_loss = float('inf')

    def log_epoch(self, epoch: int, loss: float, lr: float = None):
        """Log training epoch."""
        self.epoch_losses.append(loss)

        if loss < self.best_loss:
            self.best_loss = loss
            improvement = " (NEW BEST)"
        else:
            improvement = ""

        message = f"Epoch {epoch:3d} | Loss: {loss:.6f}{improvement}"
        if lr is not None:
            message += f" | LR: {lr:.2e}"

        self.logger.info(message)

    def log_validation(self, val_loss: float):
        """Log validation results."""
        self.logger.info(f"Validation Loss: {val_loss:.6f}")

    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping."""
        self.logger.info(
            f"Early stopping at epoch {epoch} "
            f"(patience: {patience})"
        )

    def log_training_complete(self, total_epochs: int, final_loss: float):
        """Log training completion."""
        self.logger.info(
            f"Training completed after {total_epochs} epochs. "
            f"Final loss: {final_loss:.6f} | Best loss: {self.best_loss:.6f}"
        )