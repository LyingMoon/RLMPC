import numpy as np

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