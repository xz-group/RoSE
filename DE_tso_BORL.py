"""
	This file contains code for BORL deployment
"""
import gym
import torch
from rollout.actor_critic_model_gcn_fc_rollout import ActorCritic
import numpy as np
import pickle
import networkx as nx
from scipy.linalg import fractional_matrix_power
# from tso_BORL_DE.tso_Cadence import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Check tso_BORL_DE
env_name = "gym_tso_DE:tso-v2" 
env_name_used = "PPO_39.pth"
env = gym.make(env_name)

# 6 is coming from env.observation_space.shape[0]
state_gcn_dim = 6 
# 4 specs x 16 corners + 4 normalize specs in tso.yaml
state_spec_dim = 68
action_dim = 3
# # of randomly sampled specs for deployment
num_val_specs = 200
# Episode length is maximally 50 simulations
traj_len = 50
# # of variables in hidden layer
n_latent_var = 64
# Model directory
directory = "/home/research/gaoj1/jiangao_design/trained_model/tso_BORL/"
filename = "PPO_39.pth"
# RL agent 
agent = ActorCritic(state_gcn_dim, state_spec_dim, n_latent_var, action_dim)
agent.load_state_dict(torch.load(directory + filename))
# PMOS [0, 0, 0, 0, 1]
# NMOS [0, 0, 0, 1, 0]
# Capacitor [0, 0, 1, 0, 0]
# VDD [0, 1, 0, 0, 0]
# VSS [1, 0, 0, 0, 0]
node_feature_type = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0],
                                [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
# Power node featurs
power_node_feature_para = np.array([[1.0], [0]])

def rollout(agent, env, out="assdf"):
    norm_spec_ref = env.global_g
    spec_num = len(env.specs)
    rollouts = []
    next_states = []
    obs_reached = []
    obs_nreached = []
    rollout_steps = 0
    reached_spec = 0
    # reset()
    while rollout_steps < num_val_specs:
        if out is not None:
            rollout_num = []
        state = env.reset()

        # Device parameter
        device_node_feature_para = np.reshape(state[68:78], (len(state[68:78]), 1))
        # Concatenate power node features
        node_feature_para = np.concatenate((device_node_feature_para, power_node_feature_para), axis=0)
        # Concatenate device encoding with device parameter
        state_gcn = np.concatenate((node_feature_type, node_feature_para), axis=1)
        state_spec = state[0:68]

        done = False
        reward_total = 0.0
        steps = 0
        while not done and steps < traj_len:
            action = agent.act(state_gcn, state_spec, DAD)
            states, reward, done, _ = env.step(action)

            device_node_feature_para = np.reshape(state[68:78], (len(state[68:78]), 1))
            node_feature_para = np.concatenate((device_node_feature_para, power_node_feature_para), axis=0)
            state_gcn = np.concatenate((node_feature_type, node_feature_para), axis=1)
            state_spec = states[0:68]

            reward_total += reward
            if out is not None:
                rollout_num.append(reward)
                next_states.append(states[68:78])
            steps += 1

        # Denormalize
        corner_num = 16
        norm_ideal_spec = state_spec[corner_num*spec_num:corner_num*spec_num + spec_num]
        ideal_spec = unlookup(norm_ideal_spec, norm_spec_ref)
        if done == True:
            print(done)
            reached_spec += 1
            obs_reached.append(ideal_spec)
        else:
            # Save unreached observation
            obs_nreached.append(ideal_spec)  
        if out is not None:
            rollouts.append(rollout_num)
        print("Episode reward", reward_total)
        rollout_steps += 1
        print("Specs reached: " + str(reached_spec) + "/" + str(len(obs_nreached)))
        with open('/home/research/gaoj1/jiangao_design/tso_BORL_DE/deployment.txt', 'a') as f:
            f.write("Specs reached: " + str(reached_spec) + "/" + str(len(obs_nreached)) + "\n")
    print("Num specs reached: " + str(reached_spec) + "/" + str(num_val_specs))
    print(obs_nreached)
    # end_simulator()

# Denormalization
def unlookup(norm_spec, goal_spec):
    spec = -1*np.multiply((norm_spec+1), goal_spec)/(norm_spec-1)
    return spec

if __name__ == '__main__':
    rollout(agent, env)