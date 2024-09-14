import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from NNnetwork import *
import scipy.io

class MyDataset(Dataset):
    def __init__(self,input_path,output_path):
        # Initialize the data, inside[] has to be the variable name in .mat
        input_data = scipy.io.loadmat(input_path)['INPUT']  
        output_data = scipy.io.loadmat(output_path)['OUTPUT']  
        
        self.input_data = torch.FloatTensor(input_data)
        self.output_data = torch.FloatTensor(output_data)
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        # Return a single item of the dataset
        return self.input_data[idx], self.output_data[idx]

# Initialize network, optimizer, loss function, and dataset
n_states = 10
n_hiddens = 128
n_actions = 1
action_bound = 12

input_path = r'C:\\D\\Course\\AllResearch\\QUBEServo2\\SMPCTraining\\001INPUT3.mat'
output_path = r'C:\\D\\Course\\AllResearch\\QUBEServo2\\SMPCTraining\\001OUTPUT3.mat'

net = PolicyNet(n_states, n_hiddens, n_actions, action_bound)
optimizer = optim.Adam(net.parameters(), lr=0.01) # lr= learning rate
loss_function = nn.MSELoss()  # Mean Square Error loss function
dataset = MyDataset(input_path,output_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Training loop
for epoch in range(30):  # Number of epochs
    for state, target_action in dataloader:
        optimizer.zero_grad()  # Clear gradients
        predicted_action = net(state)  # Forward pass
        loss = loss_function(predicted_action, target_action)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
    print(f'Epoch {epoch}, Loss: {loss.item()}')
    if loss.item()<0.2 and epoch>10:
        break
# Save the model
torch.save(net.state_dict(), '001SMPC4.pth')  # Saves only the model parameters

