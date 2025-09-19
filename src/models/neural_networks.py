"""Neural network architectures for RLMPC."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
import numpy as np


class PolicyNet(nn.Module):
    """Policy network for control applications.

    A feedforward neural network that maps states to actions with bounded outputs.
    Uses tanh activation for output bounding.

    Args:
        n_states: Dimension of state space
        n_hiddens: Number of hidden units
        n_actions: Dimension of action space
        action_bound: Maximum absolute value of actions
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        n_states: int,
        n_hiddens: int,
        n_actions: int,
        action_bound: float,
        dropout_rate: float = 0.0
    ) -> None:
        super(PolicyNet, self).__init__()

        if n_states <= 0 or n_hiddens <= 0 or n_actions <= 0:
            raise ValueError("Network dimensions must be positive")
        if action_bound <= 0:
            raise ValueError("Action bound must be positive")

        self.action_bound = action_bound
        self.dropout_rate = dropout_rate

        # Network layers
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(n_hiddens, n_actions)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, n_states]

        Returns:
            Action tensor of shape [batch_size, n_actions] bounded by action_bound
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.tanh(x)  # Normalize to [-1, 1]
        x = x * self.action_bound  # Scale to [-action_bound, action_bound]
        return x

    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action for a single state.

        Args:
            state: State vector
            deterministic: If True, return deterministic action

        Returns:
            Action vector
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.forward(state_tensor)

        return action.cpu().numpy().flatten()


class CriticNet(nn.Module):
    """Critic network for value function approximation.

    Q-network that estimates state-action values for DDPG.

    Args:
        n_states: Dimension of state space
        n_actions: Dimension of action space
        n_hiddens: Number of hidden units
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_hiddens: int = 128,
        dropout_rate: float = 0.0
    ) -> None:
        super(CriticNet, self).__init__()

        if n_states <= 0 or n_actions <= 0 or n_hiddens <= 0:
            raise ValueError("Network dimensions must be positive")

        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network.

        Args:
            state: State tensor [batch_size, n_states]
            action: Action tensor [batch_size, n_actions]

        Returns:
            Q-value tensor [batch_size, 1]
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
