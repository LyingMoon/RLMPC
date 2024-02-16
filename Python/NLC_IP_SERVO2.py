from asyncio.windows_events import INFINITE
import numpy as np
from quanser.hardware import *
from matplotlib import pyplot as plt
import time
from pynput import keyboard
from math import *

from Servo_2_class import *

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

qube2 = qubeservo_2(id = '0', mode = 'task')
Ts=1/(qube2.frequency) #Sample time
 
#k_p, k_d = 3, 0.14
color = np.array([0, 1, 0], dtype=np.float64)

print("Press 'space' key to stop the bot.")
print("Press enter to start the bot.")


listener =  keyboard.Listener(on_press=on_press)
listener.start()
#input = open("data.txt", "x")
#input = open("data.txt", "w")
voltage_max = 12.0
angle1 = []; angle2=[]; speed1=[];speed2=[];voltage_array=[]

desired_position = [1*sin(i/50) for i in range(0, 10000)] 
i=0
#filter
alpha=0.2

#lqr Controller
K=[-2.23606797749979,40.9843473936189,-1.80470443192922,3.46742820564671]#-2.38753267432939,3.35329489805882]

previous_position=[0,-pi]
#previous_filtered_speed1=0
previous_filtered_speed2=0
# Start the task
while break_program:
    
    if break_program:
        
        measured_position, measured_speed = qube2.read_position_and_speed()
        print('======================================')
        print('The first joint position is\n', measured_position[0])
        angle1.append(measured_position[0])
        print('The second joint position is\n', measured_position[1])
        measured_position[1]=(measured_position[1]%(2*pi))-pi
        angle2.append(measured_position[1])
        v1=measured_speed[0]
        #(measured_position[0]-previous_position[0])/Ts
        v2=(measured_position[1]-previous_position[1])/Ts
        #filtered_v1 = alpha * v1 + (1 - alpha) * previous_filtered_speed1
        filtered_v2 = alpha * v2 + (1 - alpha) * previous_filtered_speed2
        speed1.append(v1)
        speed2.append(filtered_v2)
        
        #print('The measured_speed is\n', measured_speed[0])
        print('======================================')    
        
        

        
        # Controller 
        #voltage = k_p * (measured_position[0] - desired_position[i]) #+ k_d * (measured_position[0] - desired_position[i])
        voltage=-(K[0]*(measured_position[0])+K[1]*measured_position[1]+K[2]*v1+K[3]*filtered_v2)
        
        #Save the current position
        #previous_position[0] = measured_position[0]
        previous_position[1] = measured_position[1]
        #previous_filtered_speed1 = filtered_v1
        previous_filtered_speed2 = filtered_v2
        
        # Saturation or Condition
        if abs(measured_position[1]) >= 0.2:
            voltage = 0
            
        
        if voltage >= voltage_max:
            voltage = voltage_max
            
        elif voltage <= - voltage_max:
            voltage = - voltage_max
        
        print(voltage)
        voltage_array.append(voltage)
        qube2.write_led(color)
        qube2.write_voltage(voltage)
        
        i+=1 
        print(i*Ts)
    
    
#Plot the result
qube2.terminate()
fig,  axs= plt.subplots(5)
axs[0].plot(angle1)
axs[1].plot(angle2)
axs[2].plot(speed1)
axs[3].plot(speed2)
axs[4].plot(voltage_array)
axs[0].set(xlabel= 'sample number', ylabel='joint_angle1')
axs[1].set(xlabel= 'sample number', ylabel='joint_angle2')
axs[2].set(xlabel= 'sample number', ylabel='speed1')
axs[3].set(xlabel= 'sample number', ylabel='speed2')
axs[4].set(xlabel= 'sample number', ylabel='voltage')
plt.show()

