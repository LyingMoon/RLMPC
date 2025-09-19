"""DDPG (Deep Deterministic Policy Gradient) implementation for RLMPC."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from ..models.neural_networks import PolicyNet, CriticNet
from ..config import RLMPCConfig
from ..utils.logging import TrainingLogger


@dataclass
class DDPGConfig:
    """Configuration for DDPG training."""
    buffer_size: int = 100000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005  # Soft update parameter
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    noise_std: float = 0.2
    noise_decay: float = 0.9995
    min_noise: float = 0.01
    update_frequency: int = 1
    warmup_steps: int = 1000


class ReplayBuffer:
    """Experience replay buffer for DDPG.

    Args:
        capacity: Maximum number of transitions to store
        device: Device to store tensors on
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer."""
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size {len(self.buffer)} < batch_size {batch_size}")

        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.FloatTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        )

    def size(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for action noise.

    Args:
        size: Action dimension
        mu: Mean reversion level
        theta: Mean reversion speed
        sigma: Volatility
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2
    ):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        """Reset the noise process."""
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        """Generate noise sample."""
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class DDPGAgent:
    """DDPG Agent for continuous control.

    Args:
        state_dim: State space dimension
        action_dim: Action space dimension
        action_bound: Maximum action value
        config: DDPG configuration
        device: Device for computation
        logger: Optional logger
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float,
        config: DDPGConfig = DDPGConfig(),
        device: str = "cpu",
        logger: Optional[TrainingLogger] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.config = config
        self.device = device
        self.logger = logger or TrainingLogger("DDPG")

        # Initialize networks
        self._init_networks()

        # Initialize replay buffer and noise
        self.replay_buffer = ReplayBuffer(config.buffer_size, device)
        self.noise = OrnsteinUhlenbeckNoise(
            action_dim, sigma=config.noise_std
        )

        # Training state
        self.total_steps = 0
        self.current_noise_std = config.noise_std

    def _init_networks(self) -> None:
        """Initialize actor and critic networks."""
        # Actor networks
        self.actor = PolicyNet(
            n_states=self.state_dim,
            n_hiddens=128,
            n_actions=self.action_dim,
            action_bound=self.action_bound
        ).to(self.device)

        self.actor_target = PolicyNet(
            n_states=self.state_dim,
            n_hiddens=128,
            n_actions=self.action_dim,
            action_bound=self.action_bound
        ).to(self.device)

        # Critic networks
        self.critic = CriticNet(
            n_states=self.state_dim,
            n_actions=self.action_dim,
            n_hiddens=128
        ).to(self.device)

        self.critic_target = CriticNet(
            n_states=self.state_dim,
            n_actions=self.action_dim,
            n_hiddens=128
        ).to(self.device)

        # Copy parameters to target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

    def select_action(
        self,
        state: np.ndarray,
        add_noise: bool = True,
        evaluate: bool = False
    ) -> np.ndarray:
        """Select action given state.

        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            evaluate: Whether in evaluation mode

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        if add_noise and not evaluate:
            noise = self.noise.sample() * self.current_noise_std
            action = np.clip(
                action + noise,
                -self.action_bound,
                self.action_bound
            )

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """Update actor and critic networks.

        Returns:
            Dictionary with loss values
        """
        if self.replay_buffer.size() < self.config.warmup_steps:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.config.batch_size)

        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)

        # Update actor
        actor_loss = self._update_actor(states)

        # Soft update target networks
        self.soft_update(self.actor_target, self.actor, self.config.tau)
        self.soft_update(self.critic_target, self.critic, self.config.tau)

        # Decay noise
        self.current_noise_std = max(
            self.current_noise_std * self.config.noise_decay,
            self.config.min_noise
        )

        self.total_steps += 1

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "noise_std": self.current_noise_std
        }

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Update critic network."""
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones.float()) * self.config.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, states: torch.Tensor) -> float:
        """Update actor network."""
        actions = self.actor(states)
        actor_loss = -self.critic(states, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item()

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    @staticmethod
    def hard_update(target: nn.Module, source: nn.Module) -> None:
        """Hard update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save(self, filepath: str) -> None:
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'noise_std': self.current_noise_std
        }, filepath)

        self.logger.logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        self.total_steps = checkpoint['total_steps']
        self.current_noise_std = checkpoint['noise_std']

        self.logger.logger.info(f"Agent loaded from {filepath}")

    def load_pretrained_actor(self, filepath: str) -> None:
        """Load pre-trained actor weights (e.g., from SMPC)."""
        state_dict = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.hard_update(self.actor_target, self.actor)

        self.logger.logger.info(f"Pre-trained actor loaded from {filepath}")