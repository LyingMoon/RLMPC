import csv
import torch
from NNnetwork import PolicyNet  
import scipy.io
import numpy as np
from QubeModel import *
from math import *
from matplotlib import pyplot as plt

# Test NNMPC/RLMPC in simulation

n_states=10
n_hiddens=128
n_actions=1
# When using RL + MPC, set action_bound = 12 and actor_action_bound = 3
# Otherwise set action_bound = 15 and actor_action_bound = 0
action_bound=15
actor_action_bound=0

net = PolicyNet(n_states, n_hiddens, n_actions, action_bound)
netactor = PolicyNet(n_states, n_hiddens, n_actions, actor_action_bound)

# First network is for Warm Start RL or NNMPC, second part is RL part of RL+MPC approach
# You can also use this code to test original NNMPC by disabling the actor part
net.load_state_dict(torch.load('TargetActorRealini1.pth'))
netactor.load_state_dict(torch.load('TrainActor3.pth'))
# Ensure the model is in evaluation mode
net.eval()
netactor.eval()

Time=[]
frequency=100
state=np.array([[0],[0],[0],[0]])
stateactor=np.array([[0],[0],[0],[0]])
Qube = DiscreteTimeQubeModel(state)
Qube2 = DiscreteTimeQubeModel(stateactor)
x1=[]
x2=[]
x3=[]
x4=[]
x1a=[]
x2a=[]
x3a=[]
x4a=[]
v=[]
rewardmpc=0
rewardrlmpc=0
weight = 1
feq=2
# The simulation loop
for i in range(0,701):
    Time.append(i/frequency)
    x1.extend(state[0])
    x2.extend(state[1])
    x3.extend(state[2])
    x4.extend(state[3])
    x1a.extend(stateactor[0])
    x2a.append(stateactor[1])
    x3a.append(stateactor[2])
    x4a.append(stateactor[3])
    
    NNinputstate=state.T
    Actorinputstate=stateactor.T
    # Signal 1: Sine Wave
    ref1 = np.array([weight*sin((j*0.1+(i)/frequency)*feq) for j in range(0, 6)]).reshape(1, -1)
    # Signal 2: Square Wave
    '''
    Signal = np.array([0.5,0,0,0,0,0])
    for k in range(0, 6): 
        t = np.floor((i*0.01 + 0.1*k) / np.pi)
        Signal[k] = (np.mod(t, 2) - 0.5)
    ref1=Signal.reshape(1, -1)
    '''
    NNinput = np.concatenate((NNinputstate, ref1), axis=1)
    actorinput = np.concatenate((Actorinputstate, ref1), axis=1)
   
    input_tensor = torch.tensor(np.array(NNinput), dtype=torch.float32)
    actor_input_tensor = torch.tensor(np.array(actorinput), dtype=torch.float32)
    with torch.no_grad(): 
        output = net(input_tensor)
        outputmpc = net(actor_input_tensor)
        
    with torch.no_grad(): 
        actor_output = netactor(actor_input_tensor)
        
    v1=output.squeeze().item()
    actorv1=actor_output.squeeze().item()
    newVolt=outputmpc.squeeze().item()+actorv1
    rlmpcV=np.array([[newVolt]])
    
    Input=np.array([[v1]])
    Qube.updateState(Input)
    Qube2.updateState(rlmpcV)
    
    rewardmpc = rewardmpc-5*(state[0]-weight*sin(i/frequency*feq))**2-5*(state[1])**2\
        -0.5*(Input[0,0]**2)
    rewardrlmpc = rewardrlmpc-5*(stateactor[0]-weight*sin(i/frequency*feq))**2-5*(stateactor[1])**2\
        -0.5*(rlmpcV[0,0]**2)
    
    v.append(newVolt)
    state=Qube.getState()
    stateactor=Qube2.getState()
    
    
print(rewardmpc)
print(rewardrlmpc)
'''
with open('datax1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(x1)

with open('datax2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(x2)
    
with open('datax3.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(x3)

with open('datax4.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(x4)

with open('datav1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(v)
'''
fig, axs= plt.subplots(5)
axs[0].plot(Time,x1,Time,x1a)
axs[1].plot(Time,x2,Time,x2a)
axs[2].plot(Time,x3,Time,x3a)
axs[3].plot(Time,x4,Time,x4a)
axs[4].plot(Time,v)
axs[0].set(xlabel= 'Time(s)', ylabel='x1')
axs[1].set(xlabel= 'Time(s)', ylabel='x2')
axs[2].set(xlabel= 'Time(s)', ylabel='x3')
axs[3].set(xlabel= 'Time(s)', ylabel='x4')
axs[4].set(xlabel= 'Time(s)', ylabel='v')
plt.show()
