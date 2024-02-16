from asyncio.windows_events import INFINITE
import numpy as np
from quanser.hardware import *
import time
from math import *

class qubeservo_2():
    # define read/write channels and buffer...
    w_analog_channels = np.array([0], dtype = np.uint32)
    w_digital_channels = np.array([0], dtype = np.uint32)
    w_other_channels = np.array([11000, 11001, 11002], dtype=np.uint32)
    #r_encoder_channels = np.array([0], dtype=np.uint32) # red disk
    r_encoder_channels = np.array([0, 1], dtype=np.uint32) # inverted pendulum
    r_other_channels = np.array([14000], dtype=np.uint32)
    r_digital_channels = np.array([0, 1, 2], dtype = np.uint32)

    w_num_analog_channels = len(w_analog_channels)
    w_num_digital_channels = len(w_digital_channels)
    w_num_other_channels = len(w_other_channels)
    r_num_other_channels = len(r_other_channels)
    r_num_encoder_channels = len(r_encoder_channels)
    r_num_digital_channels = len(r_digital_channels)


    w_analog_buffer = np.zeros(w_num_analog_channels, dtype=np.float64)
    w_digital_buffer = np.array([1],dtype = np.int8)
    w_other_buffer = np.zeros(w_num_other_channels, dtype=np.float64)
    r_encoder_buffer = np.zeros(r_num_encoder_channels,dtype = np.int32)
    r_other_buffer = np.zeros(r_num_other_channels, dtype=np.int32)
    r_digital_buffer = np.zeros(r_num_digital_channels, dtype=np.int8)

    #Initialization method
    def __init__(self, id = '0', mode = 'normal'):
        #Initialize the HIL object
        self.qube = HIL("qube_servo2_usb", id)
        if self.qube.is_valid():
            counts = np.array([0, 0], dtype = np.int32)
            #counts = np.array([0], dtype = np.int32)
            self.qube.set_encoder_counts(self.r_encoder_channels, self.r_num_encoder_channels, counts)
            
            #enable to write the motor
            self.qube.write_digital(self.w_digital_channels, self.w_num_digital_channels, self.w_digital_buffer)
            
            self.mode = mode 
            if mode == 'task':
                self.frequency = 500
                self.samples = INFINITE
                self.samples_in_buffer = int(self.frequency)
                self.samples_to_read = 1
                self.read_task = self.qube.task_create_reader(self.samples_in_buffer, None, 0, self.r_encoder_channels, self.r_num_encoder_channels, self.r_digital_channels, self.r_num_digital_channels, self.r_other_channels, self.r_num_other_channels)
                self.qube.task_start(self.read_task, Clock.HARDWARE_CLOCK_0, self.frequency, self.samples)
    
    # Write a voltage
    def write_voltage(self, voltage):
        self.w_analog_buffer = np.array([1 * voltage], dtype = np.float64)
        self.qube.write_analog(self.w_analog_channels, self.w_num_analog_channels, self.w_analog_buffer)
        #self.qube.task_create_analog_writer(1, self.w_analog_channels, self.w_num_analog_channels)
        print('the voltage is working')
    def write_circuit(self,input):
        self.w_digital_buffer - np.array([1 * input], dtype = np.int32)

    def read_position_and_speed(self):
        if self.mode == 'task':
            self.qube.task_read(self.read_task, self.samples_to_read, None, self.r_encoder_buffer, self.r_digital_buffer, self.r_other_buffer)
        else:
            self.qube.read(None, 0, self.r_encoder_channels, self.r_num_encoder_channels, None, 0, self.r_other_channels, self.r_num_other_channels, None, self.r_encoder_buffer, None, self.r_other_buffer)
        return -2 * np.pi *self.r_encoder_buffer/ 2048, -2 * np.pi * self.r_other_buffer / 2048
    
    def write_led(self, color):
        self.w_other_buffer = color
        self.qube.write_other(self.w_other_channels, self.w_num_other_channels, self.w_other_buffer)
        print('led is working!')

    def read_data(self):
        if self.mode =='task':
             self.qube.task_read(self.read_task, self.samples_to_read, None, self.r_encoder_buffer, self.r_digital_buffer, self.r_other_buffer)
        else:
            self.qube.read(None, 0, self.r_encoder_channels, self.r_num_encoder_channels, None, 0, self.r_other_channels, self.r_num_other_channels, None, self.r_encoder_buffer, None, self.r_other_buffer)
    
    def terminate(self):
        self.write_voltage(0)
        self.write_led(np.array([0, 1, 0], dtype = np.float64))
        self.w_digtial_buffer = np.array([0], dtype = np.int32)
        self.qube.write_digital(self.w_digital_channels, self.w_num_digital_channels, self.w_digital_buffer)
        
        if self.mode == 'task':
            self.qube.task_stop(self.read_task)
            self.qube.task_delete(self.read_task)
        self.qube.close()




