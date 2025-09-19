# On Architectures for Combining Reinforcement Learning and Model Predictive Control with Runtime Improvements

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org)
[![CVXPY](https://img.shields.io/badge/CVXPY-1.2%2B-green)](https://cvxpy.org)

**Train neural networks to replicate MPC behavior for fast real-time control.**

## Quick Start

### Installation

```bash
git clone <repository-url>
cd RLMPC
conda env create -f environment.yml
conda activate rlmpc
pip install -e .
```

### Complete Pipeline

```bash
# Step 1: Generate MPC training data
python examples/generate_training_data.py --samples 10000

# Step 2: Train Neural MPC (NNMPC) to replicate MPC behavior
python examples/generate_training_data.py --samples 10000 --train

# Step 3: Train DDPG agent (choose training mode)
python examples/train_ddpg.py --training-mode warm_start --nnmpc-path data/python_generated/neural_mpc_model.pth
```

This pipeline:
1. **Data Generation**: Collects MPC control decisions as training data
2. **NNMPC Training**: Trains neural network to replicate MPC behavior
3. **RL Training**: Three modes available for training DDPG agent:
   - **rl_only**: Train RL from scratch (±15V control range)
   - **warm_start**: Initialize RL with NNMPC weights then fine-tune (±15V control range)
   - **rl_mpc**: Hybrid approach where NNMPC provides base control (±12V) and RL adds corrections (±3V)

   **Output**: Trained actor networks are automatically saved to `results/models/actor_{mode}_{timestamp}.pth`

## Usage Examples

### Individual Steps

```bash
# Generate dataset only
python examples/generate_training_data.py --samples 5000

# Train NNMPC from existing data
python examples/generate_training_data.py --train

# Train RL agent (3 training modes available)

# Mode 1: Standard RL training from scratch
python examples/train_ddpg.py --training-mode rl_only

# Mode 2: Warm Start RL - Initialize with NNMPC weights
python examples/train_ddpg.py --training-mode warm_start --nnmpc-path data/python_generated/neural_mpc_model.pth

# Mode 3: RL+MPC Hybrid - NNMPC base control + RL corrections
python examples/train_ddpg.py --training-mode rl_mpc --nnmpc-path data/python_generated/neural_mpc_model.pth
```

### Use Different System Models
```bash
# Custom system model
python examples/generate_training_data.py --system-model custom --train

# Default Qube system (4 states)
python examples/generate_training_data.py --train
```

## Custom System Models

### Built-in Models
- `qube` - Qube Servo 2 (4 states, 1 input) - Default
- `custom` - Placeholder for custom system models

### Create Your Own System

```python
from src.mpc.python_mpc import SystemModel
import numpy as np

class MySystem(SystemModel):
    def _build_state_space(self):
        # Define: x_dot = A*x + B*u
        self.A_cont = np.array([[0, 1], [-2, -3]])
        self.B_cont = np.array([[0], [1]])
        self.A, self.B = self._c2d(self.A_cont, self.B_cont, self.Ts)
        self.C = np.eye(2)
        self.D = np.zeros((2, 1))

    @property
    def state_dim(self): return 2
    @property
    def input_dim(self): return 1
    @property
    def output_dim(self): return 2
```

Then use it:
```python
from src.utils.data_generation import DataGenerationConfig, generate_mpc_training_data

config = DataGenerationConfig(system_model_class=MySystem)
generate_mpc_training_data(config)
```
