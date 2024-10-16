import numpy as np
from math import *

class DiscreteTimeQubeModel:
    def __init__(self, initialState):
        
        self.state = np.array(initialState, dtype=float)
        # Define the A matrix
        self.A = np.array([[1,0.00729421394247470,0.00941178910802880,5.39761451023527e-07],#5.39761451023527e-07],
                        [0,1.01290901628214,-0.000581197722515543,0.0100007629895080],
                        [0,1.43078862082567,0.884659529029429,0.00258329623199078],
                        [0,2.55493625229367,-0.114004208593711,1.00449680621498]])
        # Define the B matrix
        self.B = np.array([ [0.00243062352054213],
                            [0.00240164348146919],
                            [0.476613516407320],
                            [0.471091771048391]])

    def updateState(self, controlInput):
        self.state = np.dot(self.A, self.state) + np.dot(self.B, np.array(controlInput, dtype=float))

    def getState(self):
        return self.state
    
    def step(self,controlInput,time,weight):
        Input=np.array(controlInput, dtype=float)
        Input=Input.T
        self.state = np.dot(self.A, self.state) + np.dot(self.B, np.array(Input, dtype=float))
        modelstate=self.state.T

        ref1 = np.array([weight*sin(j/10+time+0.01) for j in range(0, 6)]).reshape(1, -1)
        
        #Define the reward function here
        #reward= - 10*(modelstate[0]-ref1[0])**2 - 5*(modelstate[1]-ref2[0])**2 - 5*(modelstate[2]-ref3[0])**2
        reward = -5 * ((self.state[0, 0] - ref1[0, 0])**2) \
                 -5 * ((self.state[1, 0])**2)\
                 -0.5 * ((Input[0, 0])**2)
        RLstate= np.concatenate((modelstate, ref1), axis=1)
        isDone=False
        if abs(self.state[0,0])>1.6 or abs(self.state[1,0])>0.4:
            isDone=True
            reward=reward-1000
        return RLstate,reward,isDone,False,{}
    
    def reset(self,weight):
        self.state = np.array([[0],[0],[0],[0]])
        modelstate=self.state.T
        ref1 = np.array([weight*sin(j/10) for j in range(0, 6)]).reshape(1, -1)
        RLstate= np.concatenate((modelstate, ref1), axis=1)
        return RLstate
    