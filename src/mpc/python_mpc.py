"""Configurable MPC controller implementation in Python."""

import numpy as np
import scipy.linalg
from typing import Tuple, Optional, Dict, Any, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import cvxpy as cp
from scipy.sparse import csc_matrix


@dataclass
class MPCConfig:
    """Configuration for MPC controller."""
    prediction_horizon: int = 50  # mpcPredictStep
    sample_time: float = 0.01     # Ts
    control_sample_time: float = 0.1  # Cts
    shrinkage_steps: int = 5      # NNPredictStep

    # System model configuration
    system_model_class: Type = None  # Will default to QubeSystemModel

    # State and input dimensions (will be auto-detected from system model if not provided)
    state_dim: Optional[int] = None
    input_dim: Optional[int] = None

    # Weights
    Q: np.ndarray = None  # State weight matrix
    R: np.ndarray = None  # Input weight matrix

    # Constraints
    input_min: float = -15.0
    input_max: float = 15.0
    state_constraints: Optional[Dict[str, Tuple[float, float]]] = None

    def __post_init__(self):
        """Set default system model class if not provided."""
        if self.system_model_class is None:
            # Import here to avoid circular imports
            self.system_model_class = QubeSystemModel


class SystemModel(ABC):
    """Abstract base class for system models."""

    def __init__(self, sample_time: float = 0.01):
        """Initialize system model.

        Args:
            sample_time: Discrete sampling time in seconds
        """
        self.Ts = sample_time
        self._build_state_space()

    @abstractmethod
    def _build_state_space(self):
        """Build continuous and discrete state space matrices.

        Must set the following attributes:
        - self.A: Discrete state matrix
        - self.B: Discrete input matrix
        - self.C: Output matrix
        - self.D: Feedthrough matrix
        - self.A_cont: Continuous state matrix
        - self.B_cont: Continuous input matrix
        """
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Return the state dimension."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Return the input dimension."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension."""
        pass

    def _c2d(self, A: np.ndarray, B: np.ndarray, Ts: float) -> Tuple[np.ndarray, np.ndarray]:
        """Convert continuous to discrete time."""
        n = A.shape[0]
        m = B.shape[1]

        # Build augmented matrix
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * Ts
        M[:n, n:] = B * Ts

        # Matrix exponential
        eM = scipy.linalg.expm(M)

        Ad = eM[:n, :n]
        Bd = eM[:n, n:]

        return Ad, Bd


class QubeSystemModel(SystemModel):
    """Qube Servo 2 system model with configurable parameters."""
    
    def __init__(self, sample_time: float = 0.01):
        """Initialize Qube system model.

        Args:
            sample_time: Discrete sampling time in seconds
        """
        # Motor parameters
        self.Rm = 8.4      # Resistance
        self.kt = 0.042    # Current-torque (N-m/A)
        self.km = 0.042    # Back-emf constant (V-s/rad)

        # Rotary Arm parameters
        self.mr = 0.095    # Mass (kg)
        self.r = 0.085     # Total length (m)
        self.Jr = self.mr * self.r**2 / 3  # Moment of inertia
        self.br = 1e-3     # Viscous damping

        # Pendulum parameters
        self.mp = 0.024    # Mass (kg)
        self.Lp = 0.129    # Total length (m)
        self.l = self.Lp / 2  # Center of mass
        self.Jp = self.mp * self.Lp**2 / 3  # Moment of inertia
        self.bp = 5e-5     # Viscous damping
        self.g = 9.81      # Gravity

        # Call parent constructor
        super().__init__(sample_time)
    
    def _build_state_space(self):
        """Build continuous and discrete state space matrices."""
        # Total inertia
        Jt = self.Jr * self.Jp - self.mp**2 * self.r**2 * self.l**2
        
        # Continuous state space matrices
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, self.mp**2 * self.l**2 * self.r * self.g / Jt, 
             -self.br * self.Jp / Jt, -self.mp * self.l * self.r * self.bp / Jt],
            [0, self.mp * self.g * self.l * self.Jr / Jt, 
             -self.mp * self.l * self.r * self.br / Jt, -self.Jr * self.bp / Jt]
        ])
        
        B = np.array([[0], [0], [self.Jp / Jt], [self.mp * self.l * self.r / Jt]])
        
        # Add actuator dynamics
        A[2, 2] -= self.km**2 / self.Rm * B[2, 0]
        A[3, 2] -= self.km**2 / self.Rm * B[3, 0]
        B = self.km * B / self.Rm
        
        self.A_cont = A
        self.B_cont = B
        
        # Discretize
        self.A, self.B = self._c2d(A, B, self.Ts)
        
        # Output matrices (full state feedback)
        self.C = np.eye(4)
        self.D = np.zeros((4, 1))

    @property
    def state_dim(self) -> int:
        """Return the state dimension."""
        return 4

    @property
    def input_dim(self) -> int:
        """Return the input dimension."""
        return 1

    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return 4


class CustomSystemModel(SystemModel):
    """Placeholder for custom dynamic system models."""

    def __init__(self, sample_time: float = 0.01):
        """Initialize custom system model.

        Args:
            sample_time: Discrete sampling time in seconds
        """
        super().__init__(sample_time)

    def _build_state_space(self):
        """Placeholder - implement your custom system dynamics here."""
        raise NotImplementedError("Implement your custom system dynamics in _build_state_space()")

    @property
    def state_dim(self) -> int:
        """Return the state dimension - customize as needed."""
        raise NotImplementedError("Define state_dim for your system")

    @property
    def input_dim(self) -> int:
        """Return the input dimension - customize as needed."""
        raise NotImplementedError("Define input_dim for your system")

    @property
    def output_dim(self) -> int:
        """Return the output dimension - customize as needed."""
        raise NotImplementedError("Define output_dim for your system")


class ConfigurableMPC:
    """Configurable Model Predictive Controller."""
    
    def __init__(self, system_model: SystemModel, config: MPCConfig):
        """Initialize MPC controller.
        
        Args:
            system_model: System dynamics model
            config: MPC configuration
        """
        self.model = system_model
        self.config = config

        # Auto-detect dimensions from system model if not provided
        if config.state_dim is None:
            config.state_dim = system_model.state_dim
        if config.input_dim is None:
            config.input_dim = system_model.input_dim

        # Set default weights if not provided
        if config.Q is None:
            config.Q = np.eye(config.state_dim)
        if config.R is None:
            config.R = np.eye(config.input_dim) * 0.1
        
        # Build optimization problem
        self._setup_optimization()
    
    def _setup_optimization(self):
        """Set up the MPC optimization problem using CVXPY."""
        N = self.config.prediction_horizon
        nx = self.config.state_dim
        nu = self.config.input_dim
        
        # Decision variables
        self.x_var = cp.Variable((nx, N + 1))
        self.u_var = cp.Variable((nu, N))
        self.x0_param = cp.Parameter(nx)
        self.ref_param = cp.Parameter((nx, N))
        
        # Cost function
        cost = 0
        for k in range(N):
            # State tracking cost
            state_error = self.x_var[:, k] - self.ref_param[:, k]
            cost += cp.quad_form(state_error, self.config.Q)
            
            # Input cost
            cost += cp.quad_form(self.u_var[:, k], self.config.R)
        
        # Terminal cost
        terminal_error = self.x_var[:, N] - self.ref_param[:, N-1]
        cost += cp.quad_form(terminal_error, self.config.Q)
        
        # Constraints
        constraints = []
        
        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)
        
        # System dynamics
        for k in range(N):
            constraints.append(
                self.x_var[:, k+1] == self.model.A @ self.x_var[:, k] + 
                self.model.B @ self.u_var[:, k]
            )
        
        # Input constraints
        for k in range(N):
            constraints.append(self.u_var[:, k] >= self.config.input_min)
            constraints.append(self.u_var[:, k] <= self.config.input_max)
        
        # State constraints (if specified)
        if self.config.state_constraints:
            for k in range(N + 1):
                for i, (min_val, max_val) in enumerate(self.config.state_constraints.values()):
                    constraints.append(self.x_var[i, k] >= min_val)
                    constraints.append(self.x_var[i, k] <= max_val)
        
        # Define problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
    
    def solve(self, x0: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Solve MPC optimization problem.
        
        Args:
            x0: Initial state [4,]
            reference: Reference trajectory [4, N] or [4, N+1]
        
        Returns:
            Tuple of (optimal control input, success flag)
        """
        # Handle reference dimensions
        if reference.shape[1] == self.config.prediction_horizon + 1:
            ref = reference[:, :-1]  # Remove last column
        else:
            ref = reference
        
        # Set parameters
        self.x0_param.value = x0
        self.ref_param.value = ref
        
        # Solve
        try:
            self.problem.solve(solver=cp.OSQP, verbose=False)
            
            if self.problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return self.u_var.value[:, 0], True
            else:
                return np.zeros(self.config.input_dim), False
        except Exception as e:
            print(f"MPC solve failed: {e}")
            return np.zeros(self.config.input_dim), False
    
    def get_prediction_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the predicted state and input trajectories.
        
        Returns:
            Tuple of (predicted states [nx, N+1], predicted inputs [nu, N])
        """
        if self.problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return self.x_var.value, self.u_var.value
        else:
            return None, None


def prediction_shrinkage(reference_coarse: np.ndarray, 
                        coarse_steps: int, 
                        fine_steps: int) -> np.ndarray:
    """Apply prediction shrinkage technique to reference trajectory.
    
    Args:
        reference_coarse: Coarse reference trajectory [n_refs, coarse_steps+1]
        coarse_steps: Number of coarse prediction steps
        fine_steps: Number of fine prediction steps
    
    Returns:
        Fine reference trajectory [n_refs, fine_steps]
    """
    n_refs = reference_coarse.shape[0]
    diff = fine_steps / coarse_steps
    reference_fine = np.zeros((n_refs, fine_steps))
    
    for i in range(n_refs):
        for j in range(fine_steps):
            # Linear interpolation between coarse points
            coarse_idx = int(j / diff)
            remainder = (j % diff) / diff
            
            if coarse_idx < coarse_steps:
                reference_fine[i, j] = (
                    reference_coarse[i, coarse_idx + 1] - reference_coarse[i, coarse_idx]
                ) * remainder + reference_coarse[i, coarse_idx]
            else:
                reference_fine[i, j] = reference_coarse[i, -1]
    
    return reference_fine