from asyncio.windows_events import INFINITE
import numpy as np
from quanser.hardware import *
from matplotlib import pyplot as plt
import time
from pynput import keyboard
from math import *
from Q8_usb_class import *
from NNnetwork import PolicyNet  
import torch

n_states=36
n_hiddens=128
n_actions=4
action_bound=15

net = PolicyNet(n_states, n_hiddens, n_actions, action_bound)
break_program = True
net.load_state_dict(torch.load('002NNMPC.pth'))

net.eval()

def on_press(key):
    global break_program
    print(key)
    if key == keyboard.Key.enter and break_program:
        print ('end pressed')
        break_program = False

    if key == keyboard.Key.f1:
        print ('enter pressed')
        break_program = True

hover = Q8_usb_class(id = '0', mode = 'task')
Ts=1/(hover.frequency) #Sample time

print("Press 'space' key to stop the bot.")
print("Press enter to start the bot.")

listener =  keyboard.Listener(on_press=on_press)
listener.start()
voltage_max = 5.0

angle1 = []; angle2=[]; angle3=[]; speed1=[];speed2=[]; speed3=[]
voltage =[0,0,0,0]
voltage_array1=[]; voltage_array2=[]; voltage_array3=[]; voltage_array4=[]

#desired_position = [1*sin(i/50) for i in range(0, 10000)] 
k=0
SimulationTime=200
SimulationStep=SimulationTime/Ts
previousposition=[0,0,0]
# Start the task
while break_program and k<SimulationStep:
    if break_program and k<SimulationStep:
        measured_position, measured_speed = hover.read_position_and_speed()
        print('======================================')
        print('The yaw angle is ', measured_position[2])
        print('The pitch angle is', measured_position[0])
        print('The roll angle is ', measured_position[1])
        print('The roll speed is ', measured_speed[1])
        yaw=measured_position[2]
        roll=measured_position[1]
        pitch=measured_position[0]

        dyaw=(yaw-previousposition[2])/Ts
        droll=(roll-previousposition[1])/Ts
        dpitch=(pitch-previousposition[0])/Ts
        
        # State for the MPC: y p r dy dp dr
        state=np.array([yaw,pitch,roll,dyaw,dpitch,droll])
        # state=np.array([yaw,roll,pitch,dyaw,droll,dpitch])
        angle1.append(yaw)
        angle2.append(pitch)
        angle3.append(roll)
        speed1.append(dyaw)
        speed2.append(dpitch)
        speed3.append(droll)
        if abs(pitch)>0.4 or abs(roll)>0.4:
            break
        
        print('======================================')    
        
        # Controller 
    
        ref1 = np.array([0.3*sin(j/hover.frequency) for j in range(k, k+10)])
        #ref2 = np.array([0.15*sin(j/hover.frequency) for j in range(k, k+10)])
        #ref1 = np.array([0,0,0,0,0,0,0,0,0,0])
        ref2 = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])*0
        ref3 = np.array([0,0,0,0,0,0,0,0,0,0])
        NNMPCinput = np.concatenate((state, ref1, ref2, ref3))
        input_tensor = torch.tensor([NNMPCinput], dtype=torch.float32)
        
        with torch.no_grad(): 
            output = net(input_tensor)
        voltInput=output.squeeze()
        
        # The mpc setting: V out = Vf Vb Vr Vl
        vf=voltInput[0].item()
        vb=voltInput[1].item()
        vr=voltInput[2].item()
        vl=voltInput[3].item()
        voltage[0]=vr/3
        voltage[1]=vb/3
        voltage[2]=vl/3
        voltage[3]=vf/3
        
        # Saturation or Condition
        for i in range(0,4):
            if voltage[i] >= voltage_max:
                voltage[i] = voltage_max
            elif voltage[i] <= - voltage_max:
                voltage[i] = - voltage_max
        
        print(voltage)
        voltage_array1.append(voltage[0])
        voltage_array2.append(voltage[1])
        voltage_array3.append(voltage[2])
        voltage_array4.append(voltage[3])
        hover.write_voltage(voltage)
        
        previousposition[0]=pitch
        previousposition[1]=roll
        previousposition[2]=yaw
        k+=1 
        print(k*Ts)
    
    voltage_array=[]
#Plot the result
hover.terminate()

fig,  axs= plt.subplots(6)
axs[0].plot(angle1)
axs[1].plot(angle2)
axs[2].plot(angle3)
axs[3].plot(speed1)
axs[4].plot(speed2)
axs[5].plot(speed3)
axs[0].set(xlabel= 'sample number', ylabel='yaw angle')
axs[1].set(xlabel= 'sample number', ylabel='pitch angle')
axs[2].set(xlabel= 'sample number', ylabel='roll angle')
axs[3].set(xlabel= 'sample number', ylabel='dyaw')
axs[4].set(xlabel= 'sample number', ylabel='dpitch')
axs[5].set(xlabel= 'sample number', ylabel='droll')
plt.show()

