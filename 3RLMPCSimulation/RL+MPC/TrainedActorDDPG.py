import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random

# ------------------------------------- #
# replay buffer
# ------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):  # capacity of replay buffer
        # create a queue
        self.buffer = collections.deque(maxlen=capacity)
    # add data to the queue
    def add(self, state, action, reward, next_state, done):
        # store in the format: List
        self.buffer.append((state, action, reward, next_state, done))
    # randomly size the batch of batch size
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # divide the training set
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    # measure the current length
    def size(self):
        return len(self.buffer)

# ------------------------------------- #
# Actor Network
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions, action_bound):
        super(PolicyNet, self).__init__()
        # the maximum action
        self.action_bound = action_bound
        # only one hidden layer
        self.fc1 = nn.Linear(n_states, n_hiddens)
        
        self.fc2 = nn.Linear(n_hiddens, n_actions)
    # forward feedback
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        x= torch.tanh(x)  # [-1,1]
        x = x * self.action_bound  #  [-action_bound, action_bound]
        return x
    
        
    

# ------------------------------------- #
# Critical Network
# ------------------------------------- #

class QValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(QValueNet, self).__init__()
        # 
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)
    # Forward Network
    def forward(self, x, a):
        x = torch.squeeze(x, 1)  # This changes x from [64, 1, 36] to [64, 36]
        a = torch.squeeze(a, 1)  # This changes a from [64, 1, 4] to [64, 4]
        cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]
        x = self.fc1(cat)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # -->[b, 1]
        return x

# ------------------------------------- #
# DDPG
# ------------------------------------- #

class DDPG:
    def __init__(self, n_states, n_hiddens, n_actions, action_bound,
                 sigma, actor_lr, critic_lr, tau, gamma, device):

        # Actor Network: Training
        self.actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
        #self.actor.load_state_dict(torch.load('001SMPC2.pth'))
        # Critic Network: Target
        self.critic = QValueNet(n_states, n_hiddens, n_actions).to(device)
        #self.critic.load_state_dict(torch.load('TrainCritic2.pth'))
        # Actor Network: Target
        self.target_actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
        # Critic Network: Target
        self.target_critic = QValueNet(n_states, n_hiddens, n_actions).to(device
                                                                          )
        # Initialze the target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Initialze the target network
        self.target_actor.load_state_dict(self.actor.state_dict())

        # policy network optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # Critic network optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Settings
        self.gamma = gamma  # gamma: discount rate
        self.sigma = sigma  # gassian Noise 0
        self.tau = tau  # Soft Update parameter
        self.n_actions = n_actions
        self.device = device

    # take action
    def take_action(self, state):
        # list[n_states]-->tensor[1,n_states]-->gpu
        state = torch.tensor(state, dtype=torch.float).view(1,-1).to(self.device)
        # action [1,n_states]-->[1,1]-->int
        #action = self.actor(state).item()
        # give the action some noise: exploration
        #action = action + self.sigma * np.random.randn(self.n_actions)
        actions = self.actor(state).detach().cpu().numpy()  # Assuming the output is on GPU
        actions += self.sigma * np.random.randn(self.n_actions)  # Adding noise element-wise
        return actions
    
    # Soft Update
    def soft_update(self, net, target_net):
        # Get the updated parameters
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # update a little bit to the target network
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)

    # Train
    def update(self, transition_dict):
        # get data from the training set
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)  # [batch_size, n_actions]
        #actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,next_states]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        
        # action value [b,n_states]-->[b,n_actors]
        next_q_values = self.target_actor(next_states)
        # value [b,n_states+n_actions]-->[b,1]
        next_q_values = self.target_critic(next_states, next_q_values)
        # target q values [b,1]
        q_targets = rewards + self.gamma * next_q_values * (1-dones)
        
        # prediction [b,n_states+n_actions]-->[b,1]
        
        q_values = self.critic(states, actions)

        # the loss MSE
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # Policy Gradient
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # value for each action at current state [b, n_actions]
        actor_q_values = self.actor(states)
        # action value for current state [b,1]
        score = self.critic(states, actor_q_values)
        # calculate loss
        actor_loss = -torch.mean(score)
        # Policy Gradient
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft update for actor
        self.soft_update(self.actor, self.target_actor)
        # soft update for critic
        self.soft_update(self.critic, self.target_critic)
