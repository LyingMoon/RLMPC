# Predictive RL for MPC: Adapting to Model Parameter Variations
## Introduction
Model predictive control (MPC) is a powerful control algorithm widely used in areas like automatic driving and robotics. Based on the theoretical model, MPC solves an optimization problem while handling constraints. However, MPCrequires significant computation as the optimization needs to be performed repeatedly at each control interval. In real-world control scenarios, the theoretical model does not reflect the exact actual model dynamics, leading to suboptimal performance. In contrast with MPC’s offline design, Reinforcement Learning (RL) generates an optimal control policy through repetitive online data observation. Without knowing the actual system dynamics, RL optimizes its control policy using reward and observations from the environment [1]. This work incorporates a reinforcement learning-based model-free strategy to compensate for the issues caused by parameter variations and decrease the computational cost of MPC by using Neuron Network (NN) to approximate the MPC algorithm. By combining the online data-driven correction in RL and the offline optimization from the Neuron Network approximated Model Predictive Control (NNMPC), this proposed novel reinforcement learning-based model predictive control (RLMPC) addresses the major deficiencies of MPC.

## 1. Set up your MPC (Open 1SetupMPC)
run QubeMPCSetup.m, this code will setup the State Space Representation of the Qube Servo2 Model
