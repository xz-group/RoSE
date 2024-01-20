import torch
import gym
import sys
import time
from buffer.data_memory_gcn_fc import Memory
from model.ppo_gcn_fc import PPO
import numpy as np
import networkx as nx
from scipy.linalg import fractional_matrix_power
# from tso_BORL.tso_Cadence import *

###### generate circuits graph ######
G = nx.Graph(name='tso')
# Create nodes, considering there are 12 unique nodes in NMCF. Note that the differential structure.
G.add_node(0, name='mp1')
G.add_node(1, name='mp2')
G.add_node(2, name='mp3')
G.add_node(3, name='mp4')
G.add_node(4, name='mn1')
G.add_node(5, name='mn2')
G.add_node(6, name='mn3')
G.add_node(7, name='mn4')
G.add_node(8, name='cap1')
G.add_node(9, name='cap2')
G.add_node(10, name='vdd')
G.add_node(11, name='gnd')

# Define the edges and the edges to the graph
edges = [(0, 1), (0, 10), (1, 4), (1, 5), (2, 3), (2, 4), (2, 6), 
         (2, 8), (2, 10), (3, 4), (3, 7), (3, 8), (3, 9), (3, 10), 
         (4, 5), (4, 8), (5, 11), (6, 7), (6, 9), (6, 11), (7, 8), 
         (7, 9), (7, 11), (8, 9)]
G.add_edges_from(edges)

# See graph info
print('Graph Info:\n', nx.info(G))

# Inspect the node features
print('\nGraph Nodes: ', G.nodes.data())

# Add Self Loops
G_self_loops = G.copy()
self_loops = []
for i in range(G.number_of_nodes()):
    self_loops.append((i,i))
G_self_loops.add_edges_from(self_loops)
print('Edges of G with self-loops:\n', G_self_loops.edges)

# Get the Adjacency Matrix (A) of added self-lopps graph
A_hat = np.array(nx.attr_matrix(G_self_loops)[0])
print('Adjacency Matrix of added self-loops G (A_hat):\n', A_hat)

D = np.diag(np.sum(A_hat, axis=0))
print('Degree Matrix of added self-loops G as numpy array (D):\n', D)

# Symmetrically-normalization
D_half_norm = fractional_matrix_power(D, -0.5)
DAD = D_half_norm.dot(A_hat).dot(D_half_norm)
print('DAD:\n', DAD)

# Writing log
wtdir = "/home/research/gaoj1/jiangao_design/tso_BORL"
log = open(wtdir + '/train_gcn_fc_tso.log', 'a')
def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()
    log.flush()

############## Hyperparameters ##############
# Creating environment
# Check tso_BORL
env_name = "gym_tso:tso-v0"
env = gym.make(env_name)
# 6 is coming from env.observation_space.shape[0]
state_gcn_dim = 6
# 4 specs x 16 corners + 4 normalize specs in tso.yaml
state_spec_dim = 68
# Increase, stay the same, decrease
action_dim = 3
render = False
# If reward = 10, done
solved_reward = 10 
# Print avg reward in this interval
log_interval = 25  
# Max training episodes
max_episodes = 1000  
# Max timesteps in one episode
max_timesteps = 50 
# # of variables in hidden layer 
n_latent_var = 64  
# Update policy every n timesteps 
n_updata_episode = 30
update_timestep = n_updata_episode * max_timesteps  
lr = 0.0005
betas = (0.9, 0.9)
# Discount factor
gamma = 0.95  
# Update policy for K epochs
K_epochs = 10 
# Clip parameter for PPO 
eps_clip = 0.3  
memory = Memory()
ppo = PPO(state_gcn_dim, state_spec_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

# Logging variables
running_reward = 0
avg_length = 0
timestep = 0
niter = 0

# Training loop
mprint("Starting from Iteration %d" % niter)
# reset()
for i_episode in range(1, max_episodes + 1):
    ### one-hot encoding of gcn node features
    state = env.reset()
    # PMOS [0, 0, 0, 0, 1]
    # NMOS [0, 0, 0, 1, 0]
    # Capacitor [0, 0, 1, 0, 0]
    # VDD [0, 1, 0, 0, 0]
    # VSS [1, 0, 0, 0, 0]
    node_feature_type = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
                                  [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    # Device parameter
    device_node_feature_para = np.reshape(state[68:78], (len(state[68:78]), 1))
    # Power node featurs
    power_node_feature_para = np.array([[1.0], [0]])
    # Concatenate power node features
    node_feature_para = np.concatenate((device_node_feature_para, power_node_feature_para), axis=0)
    # Concatenate device encoding with device parameter
    state_gcn = np.concatenate((node_feature_type, node_feature_para), axis=1)
    state_spec = state[0:68]

    for t in range(max_timesteps):
        timestep += 1
            
        # Running policy_old:
        action = ppo.policy_old.act(state_gcn, state_spec, DAD, memory)
        states, reward, done, _ = env.step(action)
        # Device parameter
        device_node_feature_para = np.reshape(states[68:78], (len(states[68:78]), 1))
        # Concatenate power node features
        node_feature_para = np.concatenate((device_node_feature_para, power_node_feature_para), axis=0)
        # Concatenate device encoding with device parameter
        state_gcn = np.concatenate((node_feature_type, node_feature_para), axis=1)
        state_spec = states[0:68]

        # Saving reward and is_terminal:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        # update if its time
        if timestep % update_timestep == 0:
            ppo.update(DAD, memory)
            memory.clear_memory()
            timestep = 0

        running_reward += reward
        if render:
            env.render()
        if done:
            break

    avg_length += t

    # Stop training if avg_reward > solved_reward
    target = solved_reward

    # Logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = round(running_reward / log_interval, 3)
        mprint("[%05d] lr = %.2e, Episode=[%05d], avg length=[%02d], reward = %.2e" % (niter, lr, i_episode, avg_length, running_reward))
        torch.save(ppo.policy.state_dict(), './trained_model/tso_BORL/PPO_{}.pth'.format(niter))
        niter = niter + 1
        if running_reward > target:
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './trained_model/tso_BORL/PPO_{}.pth'.format(env_name))
            # end_simulator()
            break
        running_reward = 0
        avg_length = 0

mprint("Done!")
log.close()
# end_simulator()