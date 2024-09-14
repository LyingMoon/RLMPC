import numpy as np
import torch
import matplotlib.pyplot as plt
from TrainedActorDDPG import ReplayBuffer, DDPG
from Servo_2_class import *
from pynput import keyboard
from NNnetwork import PolicyNet  

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
break_program = True

# Keyboard Terminator (I can use this to terminate the model)
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

# -------------------------------------- #
# load the environment
# -------------------------------------- #

n_states = 10  # number of states
n_actions = 1  # number of actions
n_hiddens = 128
action_bound = 3
# action bound
mpc_action_bound = 12
batch_size=32 # Train 64 samples for each time
sigma=0 #0.4 # Gaussian Noise
tau=0.01
actor_lr=0.0001
critic_lr=0.004 #3e-3
gamma = 0.99
buffer_size=4000
min_size=10000
# -------------------------------------- #
# build the model
# -------------------------------------- #

# initialize the replay buffer
replay_buffer = ReplayBuffer(capacity=buffer_size)
# initialize the DDPG agent
agent = DDPG(n_states = n_states,  # Num of States
             n_hiddens = n_hiddens,  # hidden layers
             n_actions = n_actions,  # Num of Actions
             
             action_bound = action_bound,  # 
             sigma = sigma,  # Gaussian Noise
             actor_lr = actor_lr,  # actor learning rate
             critic_lr = critic_lr,  # critic learning rate
             tau = tau,  # soft update 
             gamma = gamma,  # gamma
             device = device
            )

mpc = PolicyNet(n_states, n_hiddens, n_actions, mpc_action_bound)
mpc.load_state_dict(torch.load('001SMPC2.pth'))
mpc.eval()
# Set up the Q8-USB
env = qubeservo_2(id = '0', mode = 'task')
Ts=1/(env.frequency) #Sample time

# -------------------------------------- #
# training the model
# -------------------------------------- #

# Initilize the parameters
voltage_max = 12.0
return_list = []  # rocord return for each 
mean_return_list = []  # average return

# Initialize the input/states
previous_position=[0,pi]
x1=0
x2=pi
statemodel = np.zeros(4, dtype=np.float64)

measured_position, measured_speed = env.read_position_and_speed()
resetvoltage =[0]
voltage =[0]

for i in range(4000):  # Iteration for n episode
    episode_return = 0  # culmulated for each episode chain
    weight=(np.random.rand()-0.5)*2
    done = False  # the episode comes to an end
    time=0 # The Timer
    ref1 = np.array([weight*sin(j*0.1) for j in range(0, 6)])
    state = np.concatenate((statemodel, ref1))
    # Reset the voltage
    env.write_voltage(resetvoltage)
    reward = 0
    # Counter
    counter = 0
    # If the system is unstable, wait until the system stabilized (manually)
    while abs(x1)>2 or abs(x2) >0.7:
        measured_position, measured_speed = env.read_position_and_speed()
        x1 = measured_position[0]
        x2 = -((measured_position[1]%(2*pi))-pi)
        x3 = (measured_position[0]-previous_position[0])/Ts
        x4 = (x2-previous_position[1])/Ts
        previous_position[0]=x1
        previous_position[1]=x2
        env.write_voltage(resetvoltage)
        counter=counter+1
        if counter%100==0:
            print('manually stablize the system')
        if counter>10000:
            break_program = True
            break
        
        
    if (np.mean(return_list[-10:]) >= -0.2) or (not break_program):
        break
        
    counter=0
    # If the system is unstable, wait until the system stabilized (mpc)
    while abs(x1)>0.1 or abs(x2) >0.1 or abs(x4)>0.1:
        measured_position, measured_speed = env.read_position_and_speed()
        x1 = measured_position[0]
        x2 = -((measured_position[1]%(2*pi))-pi)
        x3 = (measured_position[0]-previous_position[0])/Ts
        x4 = (x2-previous_position[1])/Ts
        previous_position[0]=x1
        previous_position[1]=x2
        modelstate=np.array([x1,x2,x3,x4])
        swingBackRef=np.array([0,0,0,0,0,0])
        state=np.concatenate((modelstate,swingBackRef))
        input_tensor = torch.tensor(np.array(state), dtype=torch.float32)
        with torch.no_grad(): 
            output = mpc(input_tensor)
        voltInput=output.squeeze()
        voltage= -voltInput.item()
        env.write_voltage(voltage)
        counter=counter+1
        if counter%100==0:
            print('mpc stablize the system')
        
    if (np.mean(return_list[-10:]) >= -0.2) or (not break_program):
        break
    
    measured_position, measured_speed = env.read_position_and_speed()
    x1 = measured_position[0]
    x2 = -((measured_position[1]%(2*pi))-pi)
    x3 = (measured_position[0]-previous_position[0])/Ts
    x4 = (x2-previous_position[1])/Ts
    previous_position[0]=x1
    previous_position[1]=x2
    modelstate=np.array([x1,x2,x3,x4])
    state=np.concatenate((modelstate,ref1))
    
    # For each episode, Do RL + MPC
    while not done and time<6.28:
        # what action it will take
        action = agent.take_action(state)
        input_tensor = torch.tensor(np.array(state), dtype=torch.float32)
        with torch.no_grad(): 
            output = mpc(input_tensor)
        voltInput=output.squeeze()
        voltage=-(action.item()+voltInput)
        env.write_voltage(voltage)
        
        # if the replay buffer reach its size for training, start training
        if replay_buffer.size() > min_size:
            # random sample batch_size
            s, a, r, ns, d = replay_buffer.sample(batch_size)
            # create a dataset
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # train the model
            # Store the value
            agent.update(transition_dict)
        
        # update the environment 
        measured_position, measured_speed = env.read_position_and_speed()
        x1 = measured_position[0]
        x2 = -((measured_position[1]%(2*pi))-pi)
        x3 = (measured_position[0]-previous_position[0])/Ts
        x4 = (x2-previous_position[1])/Ts

        # Check to see if the was out of constraint (Terminate Condition)
        if abs(x1)>1.5 or abs(x2) >0.4:
            print("out of constraint")
            reward=reward - 500
            done = True
        
        # Update the state
        newstate=np.array([x1,x2,x3,x4])
        ref1t = np.array([weight*sin(j*0.1+time) for j in range(0, 6)])
        next_state = np.concatenate((newstate, ref1t))
        
        # Calculate the reward
        reward = -5 * ((next_state[0] - weight*sin(time))**2) \
             -5 * ((next_state[1])**2) \
             -0.5 *((voltage)**2)
             
             
        # update the replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        # update the state
        state = next_state
        # return
        episode_return = episode_return + reward
        # update the time
        time=time+Ts
        # Update the previous position
        previous_position[1] = x2
        previous_position[0] = x1
        
        
    # save the episode return for each 
    return_list.append(episode_return)
    mean_return_list.append(np.mean(return_list[-10:])) 
    
    # print the information for this episode
    print(f'iter:{i}, return:{episode_return}, mean_return:{np.mean(return_list[-10:])}, weight:{weight}')

    # Check to see we got a policy that is good enough
    if (np.mean(return_list[-10:]) >= -4) or (not break_program) and i>10:
        break
    
env.terminate()
    
# Save the trained actor and critic
torch.save(agent.actor.state_dict(), 'TrainActorReal1.pth')  # Saves only the model parameters
torch.save(agent.target_actor.state_dict(), 'TargetActorReal1.pth') 
torch.save(agent.critic.state_dict(), 'TrainCriticReal1.pth') 
torch.save(agent.target_critic.state_dict(), 'TargetCriticReal1.pth') 

# -------------------------------------- #
# Plot
# -------------------------------------- #

x_range = list(range(len(return_list)))

plt.subplot(121)
plt.plot(x_range, return_list)  # return for every episode
plt.xlabel('episode')
plt.ylabel('return')
plt.subplot(122)
plt.plot(x_range, mean_return_list)  # average return
plt.xlabel('episode')
plt.ylabel('mean_return')
plt.show()

