"""Python data generation for MPC training, replacing MATLAB workflow."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

from typing import Type


@dataclass
class DataGenerationConfig:
    """Configuration for training data generation."""
    n_samples: int = 400000
    prediction_time: float = 0.5
    control_sample_time: float = 0.1  # Cts
    system_sample_time: float = 0.01  # Ts

    # System model configuration
    system_model_class: Type = None  # Will be set to QubeSystemModel by default

    # State ranges for randomization (will be auto-set based on system model)
    state_ranges: Dict[str, Tuple[float, float]] = None

    # Reference signal parameters
    ref_amplitude_range: Tuple[float, float] = (-1.5, 1.5)

    # Output paths
    output_dir: str = "data/generated"

    def __post_init__(self):
        """Set defaults if not provided."""
        # Import here to avoid circular imports
        from ..mpc.python_mpc import QubeSystemModel

        if self.system_model_class is None:
            self.system_model_class = QubeSystemModel

        if self.state_ranges is None:
            if self.system_model_class == QubeSystemModel:
                # Default ranges for Qube Servo 2
                self.state_ranges = {
                    'theta_arm': (-3.0, 3.0),      # x1: arm angle (larger range for first half)
                    'theta_pendulum': (-0.5, 0.5), # x2: pendulum angle
                    'omega_arm': (-8.0, 8.0),      # x3: arm angular velocity
                    'omega_pendulum': (-8.0, 8.0)  # x4: pendulum angular velocity
                }
            else:
                # Generic ranges for other systems - will be overridden based on state_dim
                # These will be set in the MPCDataGenerator based on actual dimensions
                self.state_ranges = {}


class MPCDataGenerator:
    """Generate training data for Neural MPC using Python MPC controller."""
    
    def __init__(self, config: DataGenerationConfig):
        """Initialize data generator.
        
        Args:
            config: Data generation configuration
        """
        self.config = config
        
        # Calculate prediction steps
        self.nn_predict_steps = int(config.prediction_time / config.control_sample_time)
        self.mpc_predict_steps = int(config.prediction_time / config.system_sample_time)
        self.diff = self.mpc_predict_steps / self.nn_predict_steps
        
        # Import here to avoid circular imports
        from ..mpc.python_mpc import ConfigurableMPC, MPCConfig, prediction_shrinkage

        # Initialize system model and MPC
        self.system_model = config.system_model_class(sample_time=config.system_sample_time)

        # Set up generic state ranges if not provided for non-Qube systems
        if not config.state_ranges:
            state_dim = self.system_model.state_dim
            config.state_ranges = {
                f'state_{i}': (-1.0, 1.0) for i in range(state_dim)
            }

        mpc_config = MPCConfig(
            prediction_horizon=self.mpc_predict_steps,
            sample_time=config.system_sample_time,
            control_sample_time=config.control_sample_time,
            shrinkage_steps=self.nn_predict_steps,
            system_model_class=config.system_model_class
        )

        self.mpc_controller = ConfigurableMPC(self.system_model, mpc_config)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_random_states(self) -> np.ndarray:
        """Generate random initial states for training.

        Returns:
            Random states array [N, state_dim]
        """
        N = self.config.n_samples
        state_dim = self.system_model.state_dim
        states = np.zeros((N, state_dim))

        # Generate states with different ranges
        ranges = list(self.config.state_ranges.values())

        # For small datasets, use simple uniform distribution
        if N < 100:
            for i, (min_val, max_val) in enumerate(ranges):
                states[:, i] = np.random.uniform(min_val, max_val, N)
        else:
            # Use different sampling strategies based on system type
            # Import here to avoid circular imports
            from ..mpc.python_mpc import QubeSystemModel

            if self.config.system_model_class == QubeSystemModel:
                # Qube-specific sampling strategy (mimicking MATLAB code)
                # x1 (arm angle): different ranges for first and second half
                states[:N//2, 0] = np.random.uniform(-3.0, 3.0, N//2)
                states[N//2:, 0] = np.random.uniform(-1.0, 1.0, N - N//2)

                # x2 (pendulum angle): different ranges for first and second half
                states[:N//2, 1] = np.random.uniform(ranges[1][0], ranges[1][1], N//2)
                states[N//2:, 1] = np.random.uniform(-0.4, 0.4, N - N//2)

                # x3 (arm velocity): different ranges for first quarter and rest
                states[:N//4, 2] = np.random.uniform(-8.0, 8.0, N//4)
                states[N//4:, 2] = np.random.uniform(-1.0, 1.0, N - N//4)

                # x4 (pendulum velocity): different ranges for first quarter and rest
                states[:N//4, 3] = np.random.uniform(-8.0, 8.0, N//4)
                states[N//4:, 3] = np.random.uniform(-1.0, 1.0, N - N//4)
            else:
                # Generic sampling strategy for other systems
                for i, (min_val, max_val) in enumerate(ranges):
                    # Use wider range for first half, narrower for second half
                    states[:N//2, i] = np.random.uniform(min_val, max_val, N//2)
                    states[N//2:, i] = np.random.uniform(min_val/2, max_val/2, N - N//2)

        return states
    
    def generate_reference_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate reference trajectories for training.
        
        Returns:
            Tuple of (coarse references [N, nn_steps+1], fine references [N, mpc_steps])
        """
        N = self.config.n_samples
        
        # Generate coarse reference trajectories (for neural network)
        ref_coarse = np.zeros((N, self.nn_predict_steps + 1))
        
        # Different amplitude ranges for different portions of data
        if N < 100:
            # For small datasets, use simple uniform distribution
            ref_coarse[:, :] = np.random.uniform(-1.5, 1.5, (N, self.nn_predict_steps + 1))
        else:
            ref_coarse[:N//4, :] = np.random.uniform(-3.0, 3.0, (N//4, self.nn_predict_steps + 1))
            ref_coarse[N//4:, :] = np.random.uniform(-1.0, 1.0, (N - N//4, self.nn_predict_steps + 1))
        
        # Apply prediction shrinkage to get fine references (for MPC)
        ref_fine = np.zeros((N, self.mpc_predict_steps))

        # Import here to avoid circular imports
        from ..mpc.python_mpc import prediction_shrinkage

        for i in range(N):
            # Use prediction shrinkage technique
            ref_fine[i, :] = prediction_shrinkage(
                ref_coarse[i:i+1, :],
                self.nn_predict_steps,
                self.mpc_predict_steps
            )[0, :]
        
        return ref_coarse, ref_fine
    
    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete training dataset.
        
        Returns:
            Tuple of (INPUT array [N, state_dim+nn_steps+1], OUTPUT array [N, input_dim])
        """
        self.logger.info(f"Generating {self.config.n_samples} training samples...")
        self.logger.info(f"NN prediction steps: {self.nn_predict_steps}")
        self.logger.info(f"MPC prediction steps: {self.mpc_predict_steps}")
        
        # Generate random states
        self.logger.info("Generating random initial states...")
        states = self.generate_random_states()
        
        # Generate reference trajectories
        self.logger.info("Generating reference trajectories...")
        ref_coarse, ref_fine = self.generate_reference_trajectories()
        
        # Solve MPC for each sample
        self.logger.info("Solving MPC for training samples...")
        input_dim = self.system_model.input_dim
        control_inputs = np.zeros((self.config.n_samples, input_dim))
        successful_solves = 0
        
        for i in tqdm(range(self.config.n_samples), desc="MPC Solutions"):
            # Current state and reference
            x0 = states[i, :]
            
            # Create full reference trajectory (state_dim states, only first state has reference)
            state_dim = self.system_model.state_dim
            full_ref = np.zeros((state_dim, self.mpc_predict_steps))
            full_ref[0, :] = ref_fine[i, :]  # Only first state has reference
            
            # Solve MPC
            u_opt, success = self.mpc_controller.solve(x0, full_ref)
            
            if success:
                if input_dim == 1:
                    control_inputs[i, 0] = u_opt[0]
                else:
                    control_inputs[i, :] = u_opt[:input_dim]
                successful_solves += 1
            else:
                # Fallback: simple proportional control
                if input_dim == 1:
                    control_inputs[i, 0] = np.clip(-2.0 * x0[0], -15.0, 15.0)
                else:
                    # Generic fallback for multi-input systems
                    control_inputs[i, :] = np.clip(-1.0 * x0[:input_dim], -5.0, 5.0)
        
        self.logger.info(f"Successful MPC solves: {successful_solves}/{self.config.n_samples} "
                        f"({100*successful_solves/self.config.n_samples:.1f}%)")
        
        # Combine inputs: [state (state_dim) + reference (nn_steps+1)]
        INPUT = np.hstack([states, ref_coarse])
        OUTPUT = control_inputs
        
        return INPUT, OUTPUT
    
    def save_data(self, INPUT: np.ndarray, OUTPUT: np.ndarray, 
                  prefix: str = "python_generated") -> Dict[str, Path]:
        """Save generated data as numpy arrays.
        
        Args:
            INPUT: Input data array
            OUTPUT: Output data array  
            prefix: Filename prefix
            
        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(self.config.output_dir)
        saved_files = {}
        
        # Save as numpy arrays
        input_path = output_dir / f"{prefix}_INPUT.npy"
        output_path = output_dir / f"{prefix}_OUTPUT.npy"
        
        np.save(input_path, INPUT)
        np.save(output_path, OUTPUT)
        
        saved_files['input'] = input_path
        saved_files['output'] = output_path
        
        self.logger.info(f"Saved numpy arrays: {input_path}, {output_path}")
        
        # Save generation metadata
        metadata = {
            'n_samples': self.config.n_samples,
            'nn_predict_steps': self.nn_predict_steps,
            'mpc_predict_steps': self.mpc_predict_steps,
            'prediction_time': self.config.prediction_time,
            'control_sample_time': self.config.control_sample_time,
            'system_sample_time': self.config.system_sample_time,
            'input_shape': INPUT.shape,
            'output_shape': OUTPUT.shape
        }
        
        metadata_path = output_dir / f"{prefix}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_files['metadata'] = metadata_path
        
        return saved_files


def generate_mpc_training_data(config: Optional[DataGenerationConfig] = None) -> Dict[str, Path]:
    """Convenience function to generate MPC training data.
    
    Args:
        config: Data generation configuration (uses defaults if None)
        
    Returns:
        Dictionary of saved file paths
    """
    if config is None:
        config = DataGenerationConfig()
    
    generator = MPCDataGenerator(config)
    INPUT, OUTPUT = generator.generate_training_data()
    
    return generator.save_data(INPUT, OUTPUT)


# Example usage and testing
if __name__ == "__main__":
    # Test with smaller dataset
    test_config = DataGenerationConfig(
        n_samples=1000,
        output_dir="data/test_generated"
    )
    
    saved_files = generate_mpc_training_data(test_config)
    print("Generated test data:")
    for key, path in saved_files.items():
        print(f"  {key}: {path}")