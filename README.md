# Predictive RL for MPC: Adapting to Model Parameter Variations
## Introduction
Example code and data for paper **Predictive RL for MPC: Adapting to Model Parameter Variations**

## 1. Set up your MPC
run **QubeMPCSetup.m**, this code will setup the State Space Representation of the Qube Servo2 Model

## 2. SMPC and NNMPC Training Set Generation
**001MPC4.mat** A working MPC generated by mpcDesigner in MATLAB

**001SMPC2.pth** A working NNMPC trained in Python

**AdBd001Qube.mat** The ABCD matrix of Qube-Servo2 State Space representation

**GetTrainSet001.m** The code is used to do prediction shirinkage and generate the training set for NNMPC

**MPCvsSMPC001.m** A example code comparing original MPC and SMPC (MPC with prediction shrinkage technique)




## 3. RLMPC Training in Simulation


## 4. RLMPC Training in Real World


## 5. The Result Plot
