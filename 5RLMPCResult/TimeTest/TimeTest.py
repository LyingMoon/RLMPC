import time
import torch
from NNnetwork import PolicyNet  
import scipy.io
import numpy as np
from QubeModel import *
from math import *
from matplotlib import pyplot as plt


# The test of runtime for NNMPC, Warm Start RL, and RL + MPC

n_states=10
n_hiddens=128
n_actions=1

mpc_action_bound=15

rlinimpc_action_bound=15

rlplusmpc_action_bound=12
actor_action_bound=3

netnnmpc = PolicyNet(n_states, n_hiddens, n_actions, mpc_action_bound)

netinimpc = PolicyNet(n_states, n_hiddens, n_actions, rlinimpc_action_bound)

netmpc2= PolicyNet(n_states, n_hiddens, n_actions, rlplusmpc_action_bound)
netactor = PolicyNet(n_states, n_hiddens, n_actions, actor_action_bound)

netnnmpc.load_state_dict(torch.load('001SMPC2.pth'))

netinimpc.load_state_dict(torch.load('TrainActoriniReal1.pth'))

netmpc2.load_state_dict(torch.load('001SMPC2.pth'))
netactor.load_state_dict(torch.load('TrainActorReal1.pth'))

# Ensure the model is in evaluation mode
netnnmpc.eval()
netinimpc.eval()
netmpc2.eval()
netactor.eval()

# Replace with your path
test_data_path = r'C:\\D\\Course\\AllResearch\\QubeServo2\\SMPCTraining\\TimeTestInput.mat'
test_data = scipy.io.loadmat(test_data_path)
test_input = test_data['INPUT']  # The variable name inside the .mat file
test_input_array = np.array(test_input)
test_input_tensor = torch.tensor([test_input_array], dtype=torch.float32)

timeNNi=time.time()




timeNNi=time.time()
with torch.no_grad(): 
    output = netactor(test_input_tensor)
with torch.no_grad(): 
    output = netmpc2(test_input_tensor)
timeNNf=time.time()
RLpMPCtime=timeNNf-timeNNi

timeNNi=time.time()
with torch.no_grad(): 
    output = netinimpc(test_input_tensor)

timeNNf=time.time()
RLwarmtime=timeNNf-timeNNi


timeNNi=time.time()
with torch.no_grad(): 
    output = netnnmpc(test_input_tensor)
timeNNf=time.time()
NNtime=timeNNf-timeNNi




print('Time needed for NNMPC for 400000 input set: ')
print(NNtime)
print('')
print('Time needed for Warm Start RL for 400000 input set: ')
print(RLwarmtime)
print('')
print('Time needed for RL + MPC for 400000 input set: ')
print(RLpMPCtime)
print()

p=0
for num2 in output.squeeze():
    p=p+1

print(p)