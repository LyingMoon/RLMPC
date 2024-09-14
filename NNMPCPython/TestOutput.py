import torch
from NNnetwork import PolicyNet  
import scipy.io
import numpy as np
from QubeModel import *
from math import *
from matplotlib import pyplot as plt

Time=[]
frequency=100
state=np.array([[1],[0.2],[0],[0]])
Qube = DiscreteTimeQubeModel(state)
x1=[]
x2=[]
x3=[]

k=[-2.59133129197316,42.4599733524314,-1.97840006592898,3.80059787882788]
# The simulation loop
for i in range(0,200):
    Time.append(i/frequency)
    x1.append(state[0])
    x2.append(state[1])
    x3.append(state[2])
    NNinputstate=state.T
    # Signal 1: Sine Wave
    ref1 = np.array([0*sin(j*0.1+i/frequency) for j in range(0, 6)]).reshape(1, -1)
    # Signal 2: Square Wave
    '''
    for k in range(1, 11): 
        t = np.floor((i*0.1 + 0.1*k - 0.02) / np.pi)
        Signal[k-1, 0] = (np.mod(t, 2) - 0.5)*0.4
        
    ref1=Signal.reshape(1, -1)
    '''

    v1=-(state[0,0]*k[0]+state[1,0]*k[1]+state[2,0]*k[2]+state[3,0]*k[3])

    Input=np.array([[v1]])
    Qube.updateState(Input)
    state=Qube.getState()
    

fig, axs= plt.subplots(3)
axs[0].plot(Time,x1, label='x1')
axs[1].plot(Time,x2)
axs[2].plot(Time,x3)
axs[0].set(xlabel= 'Time(s)', ylabel='x1')
axs[1].set(xlabel= 'Time(s)', ylabel='x2')
axs[2].set(xlabel= 'Time(s)', ylabel='x3')
plt.show()
