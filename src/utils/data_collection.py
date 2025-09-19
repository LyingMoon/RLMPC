"""Professional data collection system for Qube Servo 2."""

import time
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pynput import keyboard
import logging

from ..models.neural_networks import PolicyNet
from ..models.servo_interface import qubeservo_2
from ..config import RLMPCConfig
from ..utils.logging import setup_logger


@dataclass
class DataCollectionConfig:
    """Configuration for data collection experiments."""

    # Experiment parameters
    duration_seconds: float = 100.0
    start_collection_time: float = 7.0  # Start collecting after 7 seconds
    end_collection_time: float = 14.0   # Stop collecting after 14 seconds

    # Model paths
    mpc_model_path: str = "data/models/001SMPC2.pth"
    actor_model_path: str = "data/models/TrainActorReal1.pth"

    # Control parameters
    voltage_max: float = 12.0
    safety_angle_threshold: float = 0.3  # Safety cutoff angle

    # Reference signal parameters
    signal_type: int = 2  # 1=square, 2=sine
    signal_weight: float = 1.0
    signal_frequency: float = 0.1

    # Hardware parameters
    qube_id: str = "0"
    qube_mode: str = "task"

    # Data output
    save_data: bool = True
    save_plots: bool = True
    output_prefix: str = "experiment"
    results_dir: str = "results/data_collection"


@dataclass
class ExperimentData:
    """Container for collected experiment data."""

    time: List[float] = field(default_factory=list)
    joint_angle_1: List[float] = field(default_factory=list)
    joint_angle_2: List[float] = field(default_factory=list)
    joint_speed_1: List[float] = field(default_factory=list)
    joint_speed_2: List[float] = field(default_factory=list)
    voltage: List[float] = field(default_factory=list)
    reference_signal: List[float] = field(default_factory=list)
    reward: float = 0.0

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary for easy saving."""
        return {
            'time': self.time,
            'joint_angle_1': self.joint_angle_1,
            'joint_angle_2': self.joint_angle_2,
            'joint_speed_1': self.joint_speed_1,
            'joint_speed_2': self.joint_speed_2,
            'voltage': self.voltage,
            'reference_signal': self.reference_signal
        }


class KeyboardController:
    """Handle keyboard input for experiment control."""

    def __init__(self):
        self.continue_experiment = True
        self.listener = None

    def start_listening(self) -> None:
        """Start keyboard listener."""
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()

    def stop_listening(self) -> None:
        """Stop keyboard listener."""
        if self.listener:
            self.listener.stop()

    def _on_key_press(self, key) -> None:
        """Handle key press events."""
        try:
            if key == keyboard.Key.enter:
                print("Enter pressed - stopping experiment")
                self.continue_experiment = False
            elif key == keyboard.Key.f1:
                print("F1 pressed - continuing experiment")
                self.continue_experiment = True
        except AttributeError:
            pass  # Special keys


class ReferenceSignalGenerator:
    """Generate reference signals for tracking experiments."""

    def __init__(self, config: DataCollectionConfig):
        self.config = config

    def generate_signal(self, current_time: float, prediction_horizon: int = 6) -> np.ndarray:
        """Generate reference signal for current time and prediction horizon.

        Args:
            current_time: Current experiment time
            prediction_horizon: Number of future time steps to predict

        Returns:
            Array of reference values for current and future time steps
        """
        signal = np.zeros(prediction_horizon)

        for i in range(prediction_horizon):
            future_time = current_time + i * 0.1  # 0.1s prediction steps

            if self.config.signal_type == 1:  # Square wave
                signal[i] = self.config.signal_weight * (
                    (int(future_time / np.pi) % 2) - 0.5
                )
            elif self.config.signal_type == 2:  # Sine wave
                signal[i] = self.config.signal_weight * np.sin(
                    self.config.signal_frequency * future_time
                )
            else:
                raise ValueError(f"Unknown signal type: {self.config.signal_type}")

        return signal


class QubeDataCollector:
    """Professional data collection system for Qube Servo 2."""

    def __init__(
        self,
        config: DataCollectionConfig = DataCollectionConfig(),
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or setup_logger("QubeDataCollector")

        # Initialize components
        self.keyboard_controller = KeyboardController()
        self.signal_generator = ReferenceSignalGenerator(config)
        self.experiment_data = ExperimentData()

        # Initialize models and hardware
        self._initialize_models()
        self._initialize_hardware()

        # State tracking
        self.previous_position = np.array([0.0, np.pi])
        self.step_count = 0

    def _initialize_models(self) -> None:
        """Initialize neural network models."""
        try:
            # Load MPC model
            self.mpc_model = PolicyNet(
                n_states=10,  # 4 states + 6 reference signals
                n_hiddens=128,
                n_actions=1,
                action_bound=15.0
            )
            self.mpc_model.load_state_dict(
                torch.load(self.config.mpc_model_path, map_location='cpu')
            )
            self.mpc_model.eval()

            # Load actor model
            self.actor_model = PolicyNet(
                n_states=10,
                n_hiddens=128,
                n_actions=1,
                action_bound=0.0  # Actor provides adjustment
            )
            self.actor_model.load_state_dict(
                torch.load(self.config.actor_model_path, map_location='cpu')
            )
            self.actor_model.eval()

            self.logger.info("Models loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise

    def _initialize_hardware(self) -> None:
        """Initialize Qube Servo 2 hardware."""
        try:
            self.qube = qubeservo_2(
                id=self.config.qube_id,
                mode=self.config.qube_mode
            )
            self.sample_time = 1.0 / self.qube.frequency
            self.logger.info(f"Qube initialized with sample time: {self.sample_time:.4f}s")

        except Exception as e:
            self.logger.error(f"Failed to initialize Qube hardware: {e}")
            raise

    def _read_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read current state from Qube hardware.

        Returns:
            Tuple of (position, velocity) arrays
        """
        measured_position, _ = self.qube.read_position_and_speed()

        # Process angles
        angle_1 = measured_position[0]
        angle_2 = -((measured_position[1] % (2 * np.pi)) - np.pi)

        # Calculate velocities
        velocity_1 = (angle_1 - self.previous_position[0]) / self.sample_time
        velocity_2 = (angle_2 - self.previous_position[1]) / self.sample_time

        return np.array([angle_1, angle_2]), np.array([velocity_1, velocity_2])

    def _compute_control(self, state: np.ndarray, reference: np.ndarray) -> float:
        """Compute control input using neural networks.

        Args:
            state: Current state [angle1, angle2, velocity1, velocity2]
            reference: Reference signal for prediction horizon

        Returns:
            Control voltage
        """
        # Prepare input for neural networks
        nn_input = np.concatenate([state, reference])
        input_tensor = torch.tensor(nn_input, dtype=torch.float32)

        # Get outputs from both models
        with torch.no_grad():
            mpc_output = self.mpc_model(input_tensor)
            actor_output = self.actor_model(input_tensor)

        # Combine outputs
        voltage = -(mpc_output.item() + actor_output.item())

        # Apply safety constraints
        if abs(state[1]) >= self.config.safety_angle_threshold:
            voltage = 0.0
            self.logger.warning("Safety cutoff activated - large angle detected")

        # Apply voltage limits
        voltage = np.clip(voltage, -self.config.voltage_max, self.config.voltage_max)

        return voltage

    def _calculate_reward(self, state: np.ndarray, reference: float, voltage: float) -> float:
        """Calculate reward for current step."""
        tracking_error = (state[0] - reference) ** 2
        stabilization_error = state[1] ** 2
        control_penalty = voltage ** 2

        return -5.0 * tracking_error - 5.0 * stabilization_error - 0.5 * control_penalty

    def _should_collect_data(self, current_time: float) -> bool:
        """Determine if we should collect data at current time."""
        return (self.config.start_collection_time <= current_time <=
                self.config.end_collection_time)

    def run_experiment(self) -> ExperimentData:
        """Run the data collection experiment.

        Returns:
            Collected experiment data
        """
        self.logger.info("Starting data collection experiment")
        self.keyboard_controller.start_listening()

        # LED color for visual feedback
        led_color = np.array([0, 1, 0], dtype=np.float64)  # Green

        try:
            while (self.keyboard_controller.continue_experiment and
                   self.step_count * self.sample_time < self.config.duration_seconds):

                current_time = self.step_count * self.sample_time

                # Read current state
                position, velocity = self._read_state()
                state = np.concatenate([position, velocity])

                # Generate reference signal
                reference = self.signal_generator.generate_signal(current_time)

                # Compute control
                voltage = self._compute_control(state, reference)

                # Apply control
                self.qube.write_led(led_color)
                self.qube.write_voltage(voltage)

                # Collect data if in collection window
                if self._should_collect_data(current_time):
                    self.experiment_data.time.append(current_time)
                    self.experiment_data.joint_angle_1.append(position[0])
                    self.experiment_data.joint_angle_2.append(position[1])
                    self.experiment_data.joint_speed_1.append(velocity[0])
                    self.experiment_data.joint_speed_2.append(velocity[1])
                    self.experiment_data.voltage.append(voltage)
                    self.experiment_data.reference_signal.append(reference[0])

                    # Update reward
                    step_reward = self._calculate_reward(state, reference[0], voltage)
                    self.experiment_data.reward += step_reward

                # Update state tracking
                self.previous_position = position.copy()
                self.step_count += 1

                # Log progress periodically
                if self.step_count % 100 == 0:
                    self.logger.info(f"Step {self.step_count}, Time: {current_time:.2f}s")

        except KeyboardInterrupt:
            self.logger.info("Experiment interrupted by user")
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise
        finally:
            self._cleanup()

        self.logger.info(f"Experiment completed. Total reward: {self.experiment_data.reward:.2f}")
        return self.experiment_data

    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.qube.terminate()
            self.keyboard_controller.stop_listening()
            self.logger.info("Hardware and keyboard listener cleaned up")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def save_data(self, data: ExperimentData, suffix: str = "") -> None:
        """Save experiment data to CSV files."""
        if not self.config.save_data:
            return

        # Create output directory
        output_dir = Path(self.config.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each data series
        data_dict = data.to_dict()
        for name, values in data_dict.items():
            if values:  # Only save non-empty data
                filename = f"{self.config.output_prefix}_{name}{suffix}.csv"
                filepath = output_dir / filename

                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(values)

        self.logger.info(f"Data saved to {output_dir}")

    def create_plots(self, data: ExperimentData, suffix: str = "") -> None:
        """Create and save plots of experiment data."""
        if not self.config.save_plots or not data.time:
            return

        fig, axes = plt.subplots(5, 1, figsize=(12, 10))
        fig.suptitle(f'Qube Servo 2 Experiment Results{suffix}')

        # Joint angle 1 with reference
        axes[0].plot(data.time, data.joint_angle_1, label='Actual', linewidth=2)
        axes[0].plot(data.time, data.reference_signal, label='Reference',
                    linestyle='--', linewidth=2)
        axes[0].set_ylabel('Joint Angle 1 (rad)')
        axes[0].legend()
        axes[0].grid(True)

        # Joint angle 2
        axes[1].plot(data.time, data.joint_angle_2, linewidth=2)
        axes[1].set_ylabel('Joint Angle 2 (rad)')
        axes[1].grid(True)

        # Joint speed 1
        axes[2].plot(data.time, data.joint_speed_1, linewidth=2)
        axes[2].set_ylabel('Joint Speed 1 (rad/s)')
        axes[2].grid(True)

        # Joint speed 2
        axes[3].plot(data.time, data.joint_speed_2, linewidth=2)
        axes[3].set_ylabel('Joint Speed 2 (rad/s)')
        axes[3].grid(True)

        # Voltage
        axes[4].plot(data.time, data.voltage, linewidth=2)
        axes[4].set_ylabel('Voltage (V)')
        axes[4].set_xlabel('Time (s)')
        axes[4].grid(True)

        plt.tight_layout()

        if self.config.save_plots:
            output_dir = Path(self.config.results_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_file = output_dir / f"{self.config.output_prefix}_plots{suffix}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plots saved to {plot_file}")

        plt.show()


def run_data_collection_experiment(
    config: Optional[DataCollectionConfig] = None,
    suffix: str = ""
) -> ExperimentData:
    """Convenience function to run a data collection experiment.

    Args:
        config: Experiment configuration
        suffix: Suffix for output files

    Returns:
        Collected experiment data
    """
    if config is None:
        config = DataCollectionConfig()

    collector = QubeDataCollector(config)

    try:
        data = collector.run_experiment()
        collector.save_data(data, suffix)
        collector.create_plots(data, suffix)
        return data

    except Exception as e:
        collector.logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    config = DataCollectionConfig(
        duration_seconds=20.0,
        signal_type=2,  # Sine wave
        signal_weight=1.0,
        output_prefix="rlmpc_experiment"
    )

    print("Starting Qube Servo 2 data collection...")
    print("Press Enter to stop, F1 to continue")

    try:
        data = run_data_collection_experiment(config)
        print(f"Experiment completed! Total reward: {data.reward:.2f}")
        print(f"Collected {len(data.time)} data points")

    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()