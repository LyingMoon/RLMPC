#!/usr/bin/env python3
"""Example script for training DDPG agent."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.mpc.python_mpc import QubeSystemModel, CustomSystemModel
from src.rl.ddpg import DDPGAgent, DDPGConfig
from src.utils.logging import TrainingLogger, get_timestamp
from src.models.neural_networks import PolicyNet
import torch


class SystemSimulator:
    """Generic simulator wrapper for any SystemModel."""

    def __init__(self, system_model, initial_state: np.ndarray, sample_time: float = 0.01):
        self.system = system_model(sample_time)
        self.state = np.array(initial_state, dtype=float)

    def getState(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()

    def updateState(self, action: np.ndarray) -> None:
        """Update state with control action."""
        # Ensure action is correct shape
        if np.isscalar(action):
            action = np.array([action])
        elif action.ndim > 1:
            action = action.flatten()

        # Simulate one step: x_{k+1} = A*x_k + B*u_k
        if self.system.input_dim == 1:
            self.state = self.system.A @ self.state + self.system.B.flatten() * action[0]
        else:
            self.state = self.system.A @ self.state + self.system.B @ action[:self.system.input_dim]


class RLMPCEnvironment:
    """Configurable environment wrapper for RL training."""

    def __init__(self, system_model_class=QubeSystemModel, target_state: np.ndarray = None, sample_time: float = 0.01):
        # Create instance to get dimensions
        temp_system = system_model_class(sample_time)
        state_dim = temp_system.state_dim

        # Initialize with zero state
        initial_state = np.zeros(state_dim)
        self.simulator = SystemSimulator(system_model_class, initial_state, sample_time)

        # Set target state based on system type
        if target_state is not None:
            self.target_state = target_state
        elif system_model_class == QubeSystemModel:
            self.target_state = np.array([0, 0, np.pi, 0])  # Inverted pendulum
        else:
            self.target_state = np.zeros(state_dim)  # Generic zero target

        self.max_steps = 1000
        self.current_step = 0
        self.state_dim = state_dim

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        initial_state = np.random.normal(0, 0.1, self.state_dim)  # Small random initialization
        self.simulator.state = initial_state
        self.current_step = 0
        return self.simulator.getState()

    def step(self, action: np.ndarray) -> tuple:
        """Execute action and return next state, reward, done."""
        # Apply action
        self.simulator.updateState(action)
        next_state = self.simulator.getState()

        # Calculate reward (negative squared distance to target)
        error = next_state - self.target_state
        reward = -np.sum(error ** 2) - 0.01 * np.sum(action ** 2)  # Add action penalty

        # Check if done
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Check for instability (large angle deviation)
        if abs(next_state[2]) > 2 * np.pi or abs(next_state[0]) > 1.0:
            reward -= 100  # Large penalty for instability
            done = True

        return next_state, reward, done


def train_ddpg_agent(system_model_class=QubeSystemModel, training_mode="rl_only", nnmpc_path=None, num_episodes=1000):
    """Train DDPG agent on configurable environment."""

    # Configuration based on training mode
    if training_mode == "rl_mpc":
        # RL+MPC mode: lower action bounds for RL corrections
        action_bound = 3.0  # RL provides ±3V corrections
        config = DDPGConfig(
            buffer_size=100000,
            batch_size=64,
            gamma=0.99,
            tau=0.005,
            actor_lr=0.0001,
            critic_lr=0.001,
            noise_std=0.2,  # Lower noise for corrections
            noise_decay=0.9995,
            min_noise=0.01,
            warmup_steps=1000
        )
    else:
        # rl_only or warm_start modes: full voltage range
        action_bound = 15.0  # Full ±15V range
        config = DDPGConfig(
            buffer_size=100000,
            batch_size=64,
            gamma=0.99,
            tau=0.005,
            actor_lr=0.0001,
            critic_lr=0.001,
            noise_std=0.3,
            noise_decay=0.9995,
            min_noise=0.05,
            warmup_steps=1000
        )

    # Environment and agent
    env = RLMPCEnvironment(system_model_class=system_model_class)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Set up logging
    timestamp = get_timestamp()
    logger = TrainingLogger("DDPG", f"results/logs/ddpg_{timestamp}.log")

    # Get system dimensions
    temp_system = system_model_class()
    state_dim = temp_system.state_dim
    input_dim = temp_system.input_dim

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=input_dim,
        action_bound=action_bound,
        config=config,
        device=device,
        logger=logger
    )

    # Load NNMPC model for warm_start or rl_mpc modes
    nnmpc_model = None
    if training_mode in ["warm_start", "rl_mpc"] and nnmpc_path:
        if Path(nnmpc_path).exists():
            try:
                nnmpc_model = PolicyNet(
                    n_states=state_dim + 6,  # 4 states + 6 reference points (from downsampling)
                    n_hiddens=128,
                    n_actions=input_dim,
                    action_bound=12.0 if training_mode == "rl_mpc" else 15.0
                )
                nnmpc_model.load_state_dict(torch.load(nnmpc_path, map_location=device))
                nnmpc_model.to(device)
                nnmpc_model.eval()
                logger.logger.info(f"Loaded NNMPC model from {nnmpc_path}")

                # For warm_start mode, initialize actor with NNMPC weights (compatible layers only)
                if training_mode == "warm_start":
                    try:
                        # Load compatible weights from NNMPC (skip incompatible layers)
                        actor_state_dict = agent.actor.state_dict()
                        nnmpc_state_dict = nnmpc_model.state_dict()

                        # Only load compatible layers (skip fc1 due to input size mismatch)
                        compatible_keys = []
                        for key in actor_state_dict.keys():
                            if key in nnmpc_state_dict and actor_state_dict[key].shape == nnmpc_state_dict[key].shape:
                                actor_state_dict[key] = nnmpc_state_dict[key]
                                compatible_keys.append(key)

                        agent.actor.load_state_dict(actor_state_dict)
                        agent.hard_update(agent.actor_target, agent.actor)
                        logger.logger.info(f"Warm start: Loaded compatible layers: {compatible_keys}")
                    except Exception as e:
                        logger.logger.warning(f"Failed to warm start actor: {e}")

            except Exception as e:
                logger.logger.warning(f"Failed to load NNMPC model: {e}")
                nnmpc_model = None
        else:
            logger.logger.warning(f"NNMPC model not found at {nnmpc_path}")

    # Fallback to old behavior for compatibility
    elif training_mode == "warm_start":
        fallback_path = "data/models/001SMPC2.pth"
        if Path(fallback_path).exists():
            try:
                agent.load_pretrained_actor(fallback_path)
                logger.logger.info("Loaded fallback pre-trained actor")
            except Exception as e:
                logger.logger.warning(f"Failed to load fallback actor: {e}")

    # Training parameters
    save_frequency = 100
    eval_frequency = 50

    # Training loop
    episode_rewards = []
    best_reward = -float('inf')

    logger.logger.info(f"Starting DDPG training for {num_episodes} episodes")
    logger.logger.info(f"Device: {device}")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            # Select action based on training mode
            if training_mode == "rl_mpc" and nnmpc_model is not None:
                # RL+MPC mode: combine NNMPC output with RL correction
                with torch.no_grad():
                    # Create reference trajectory (simplified for now)
                    reference = np.array([env.target_state[0]] * 6)  # 6 reference points for position
                    nnmpc_input = torch.FloatTensor(np.concatenate([state, reference])).unsqueeze(0).to(device)
                    nnmpc_action = nnmpc_model(nnmpc_input).cpu().numpy().flatten()

                # Get RL correction
                rl_correction = agent.select_action(state, add_noise=True)

                # Combine: NNMPC + RL correction
                action = nnmpc_action + rl_correction
                action = np.clip(action, -15.0, 15.0)  # Final voltage constraint
            else:
                # Standard RL training (rl_only or warm_start modes)
                action = agent.select_action(state, add_noise=True)

            # Execute action
            next_state, reward, done = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            if step_count % config.update_frequency == 0:
                losses = agent.update()

            state = next_state
            episode_reward += reward
            step_count += 1

            if done:
                break

        episode_rewards.append(episode_reward)

        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.logger.info(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Avg(10): {avg_reward:8.2f} | "
                f"Noise: {agent.current_noise_std:.4f} | "
                f"Buffer: {agent.replay_buffer.size()}"
            )

        # Evaluation
        if episode % eval_frequency == 0 and episode > 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=5)
            logger.logger.info(f"Evaluation reward: {eval_reward:.2f}")

            if eval_reward > best_reward:
                best_reward = eval_reward
                save_path = f"data/models/ddpg_best_{timestamp}.pth"
                agent.save(save_path)
                logger.logger.info(f"New best model saved: {eval_reward:.2f}")

        # Save checkpoint
        if episode % save_frequency == 0 and episode > 0:
            save_path = f"data/models/ddpg_episode_{episode}_{timestamp}.pth"
            agent.save(save_path)

    logger.logger.info("Training completed!")
    logger.logger.info(f"Best evaluation reward: {best_reward:.2f}")

    # Save final actor network for deployment
    timestamp = get_timestamp()
    actor_save_path = f"results/models/actor_{training_mode}_{timestamp}.pth"

    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(actor_save_path), exist_ok=True)

    # Save actor network state dict
    torch.save(agent.actor.state_dict(), actor_save_path)
    logger.logger.info(f"Final actor network saved to {actor_save_path}")

    return agent, episode_rewards


def evaluate_agent(agent: DDPGAgent, env: RLMPCEnvironment, num_episodes: int = 5) -> float:
    """Evaluate agent performance."""
    total_reward = 0

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            # Select action for evaluation
            if training_mode == "rl_mpc" and nnmpc_model is not None:
                # RL+MPC mode: combine NNMPC output with RL correction
                with torch.no_grad():
                    reference = np.array([env.target_state[0]] * 6)  # 6 reference points for position
                    nnmpc_input = torch.FloatTensor(np.concatenate([state, reference])).unsqueeze(0).to(device)
                    nnmpc_action = nnmpc_model(nnmpc_input).cpu().numpy().flatten()

                rl_correction = agent.select_action(state, add_noise=False, evaluate=True)
                action = nnmpc_action + rl_correction
                action = np.clip(action, -15.0, 15.0)
            else:
                action = agent.select_action(state, add_noise=False, evaluate=True)
            next_state, reward, done = env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                break

        total_reward += episode_reward

    return total_reward / num_episodes


def main():
    """Main training function with configurable system model."""
    import argparse

    parser = argparse.ArgumentParser(description="Train DDPG agent on configurable system")
    parser.add_argument("--system-model", type=str, default="qube",
                       choices=["qube", "custom"],
                       help="System model to use (default: qube)")
    parser.add_argument("--training-mode", type=str, default="rl_only",
                       choices=["rl_only", "warm_start", "rl_mpc"],
                       help="Training mode: rl_only (scratch), warm_start (init with NNMPC), rl_mpc (hybrid)")
    parser.add_argument("--nnmpc-path", type=str, default="results/models/neural_mpc.pth",
                       help="Path to pre-trained NNMPC model for initialization")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes (default: 1000)")

    args = parser.parse_args()

    # Map system model choice to class
    system_models = {
        "qube": QubeSystemModel,
        "custom": CustomSystemModel
    }

    selected_model = system_models[args.system_model]
    print(f"Training DDPG agent on {args.system_model} system")
    print(f"Training mode: {args.training_mode}")

    try:
        agent, rewards = train_ddpg_agent(
            system_model_class=selected_model,
            training_mode=args.training_mode,
            nnmpc_path=args.nnmpc_path,
            num_episodes=args.episodes
        )
        print(f"Training completed. Final average reward: {np.mean(rewards[-100:]):.2f}")

        # The actor network path is available in the agent training function logs
        print("Check logs for actor network save location.")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()