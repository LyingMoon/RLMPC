from asyncio.windows_events import INFINITE
import numpy as np
from quanser.hardware import *
from matplotlib import pyplot as plt
import time
from pynput import keyboard
from math import *
import torch
from NNnetwork import PolicyNet  
from Servo_2_class import *
import math
import csv
n_states=10
n_hiddens=128
n_actions=1
mpc_action_bound=15
actor_action_bound=0

net = PolicyNet(n_states, n_hiddens, n_actions, mpc_action_bound)
netactor = PolicyNet(n_states, n_hiddens, n_actions, actor_action_bound)
net.load_state_dict(torch.load('001SMPC2.pth'))
netactor.load_state_dict(torch.load('TrainActorReal1.pth'))
# Ensure the model is in evaluation mode
net.eval()
netactor.eval()
rewardmpc=0
break_program = True

def on_press(key):
    global break_program
    print(key)
    if key == keyboard.Key.enter and break_program:
        print ('end pressed')
        break_program = False
    if key == keyboard.Key.f1:
        print ('enter pressed')
        break_program = True
listener =  keyboard.Listener(on_press=on_press)
listener.start()

qube2 = qubeservo_2(id = '0', mode = 'task')
Ts=1/(qube2.frequency) #Sample time
color = np.array([0, 1, 0], dtype=np.float64)

voltage_max = 12.0
angle1 = []; angle2=[]; speed1=[];speed2=[];voltage_array=[]

desired_position = [1*sin(i/100) for i in range(0, 10000)] 

previous_position=[0,pi]
Signal=[]
timeplot=[]
# initialize the counter

weight = 1

i = 0
SignalType=2
while break_program and i <= 10000:
    measured_position, measured_speed = qube2.read_position_and_speed()
    
    #print('The first joint position is\n', measured_position[0])
    #print('The second joint position is\n', measured_position[1])
    
    x1 = measured_position[0]
    x2 = -((measured_position[1]%(2*pi))-pi)
    x3 = (measured_position[0]-previous_position[0])/Ts
    x4 = (x2-previous_position[1])/Ts
    
    ref1 = np.array([weight*sin(j*0.1+i*Ts) for j in range(0, 6)])
    
    if SignalType ==1:
        for j in range(0,6):
            ref1[j] = weight*(math.floor((j*0.1+i*Ts)/np.pi)%2-0.5)
        #Signal.append(math.floor((i*Ts)/np.pi)%2-0.5)
    
    
    # Controller 
    NNinputstate = np.array([x1,x2,x3,x4])
    NNinput = np.concatenate((NNinputstate, ref1))
    input_tensor = torch.tensor(np.array(NNinput), dtype=torch.float32)
    with torch.no_grad(): 
        output = net(input_tensor)
    with torch.no_grad(): 
        outputactor = netactor(input_tensor)
    voltInput=output.squeeze()
    actorInput=outputactor.squeeze()
    voltage=-(voltInput.item()+actorInput.item())
    
    # Saturation or Condition
    if abs(x2) >= 0.3:
        voltage = 0
    
    if voltage >= voltage_max:
        voltage = voltage_max
        
    elif voltage <= - voltage_max:
        voltage = - voltage_max
    
    print(voltage)
    if i>=700 and i <1400:
        voltage_array.append(voltage)
        angle1.append(x1)
        angle2.append(x2)
        speed1.append(x3)
        speed2.append(x4)
        timeplot.append(i*Ts)
        Signal.append(ref1[0])
        rewardmpc = rewardmpc-5*(x1-ref1[0])**2-5*(x2)**2\
        -0.5*(voltage**2)
        
    
    qube2.write_led(color)
    qube2.write_voltage(voltage)
    
    #Save the current position
    previous_position[0] = x1
    previous_position[1] = x2
    
    i+=1 
    print(i*Ts)
    
print(rewardmpc)
    
#Plot the result
qube2.terminate()

'''
with open('nnmpcx1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(angle1)

with open('nnmpcx2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(angle2)
    
with open('nnmpcx3.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(speed1)

with open('nnmpcx4.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(speed2)

with open('nnmpcv1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(voltage_array)
'''

fig,  axs= plt.subplots(5)
axs[0].plot(timeplot,angle1,timeplot,Signal)
axs[1].plot(timeplot,angle2)
axs[2].plot(timeplot,speed1)
axs[3].plot(timeplot,speed2)
axs[4].plot(voltage_array)
axs[0].set(xlabel= 'sample number', ylabel='joint_angle1')
axs[1].set(xlabel= 'sample number', ylabel='joint_angle2')
axs[2].set(xlabel= 'sample number', ylabel='speed1')
axs[3].set(xlabel= 'sample number', ylabel='speed2')
axs[4].set(xlabel= 'sample number', ylabel='voltage')
plt.show()

