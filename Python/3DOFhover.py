from asyncio.windows_events import INFINITE
import numpy as np
from quanser.hardware import *
from matplotlib import pyplot as plt
import time
from pynput import keyboard
from math import *

from Q8_usb_class import *

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

hover = Q8_usb_class(id = '0', mode = 'task')
Ts=1/(hover.frequency) #Sample time

print("Press 'space' key to stop the bot.")
print("Press enter to start the bot.")

listener =  keyboard.Listener(on_press=on_press)
listener.start()
voltage_max = 15.0

angle1 = []; angle2=[]; angle3=[]; speed1=[];speed2=[]; speed3=[]
voltage_array1=[]; voltage_array2=[]; voltage_array3=[]; voltage_array4=[]

desired_position = [1*sin(i/50) for i in range(0, 10000)] 
i=0
alpha=0.2

#lqr Controller
K=[-2.23606797749979,40.9843473936189,-1.80470443192922,3.46742820564671]

# Start the task
while break_program:
    
    if break_program:
        
        measured_position, measured_speed = hover.read_position_and_speed()
        print('======================================')
        print('The first joint position is\n', measured_position[0])
        print('The second joint position is\n', measured_position[1])
        angle1.append(measured_position[0])
        angle2.append(measured_position[1])
        angle3.append(measured_position[2])
        
        speed1.append(measured_speed[0])
        speed2.append(measured_speed[1])
        speed3.append(measured_speed[2])
        
        print('======================================')    
        
        

        
        # Controller 
        voltage1=0
        voltage2=0
        voltage3=0
        voltage4=0
        
        # Saturation or Condition
        if abs(measured_position[1]) >= 0.2:
            voltage = 0
            
        if voltage >= voltage_max:
            voltage = voltage_max
            
        elif voltage <= - voltage_max:
            voltage = - voltage_max
        
        print(voltage)
        voltage_array1.append(voltage1)
        voltage_array1.append(voltage1)
        voltage_array1.append(voltage1)
        voltage_array1.append(voltage1)
        hover.write_voltage(voltage)
        
        i+=1 
        print(i*Ts)
    
    
#Plot the result
hover.terminate()
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

