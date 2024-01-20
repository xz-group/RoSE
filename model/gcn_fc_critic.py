"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import numpy as np

class GCN_FC_critic(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim_gcn, in_dim_spec, n_latent_var, out_dim):
		"""
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""
		super(GCN_FC_critic, self).__init__()


		self.layer_gcn_1 = nn.Linear(in_dim_gcn, n_latent_var)
		self.layer_gcn_2 = nn.Linear(n_latent_var, n_latent_var)
		self.layer_gcn_3 = nn.Linear(n_latent_var, 4)

		self.layer_fc_1 = nn.Linear(in_dim_spec, n_latent_var)
		# 12 = # of device + vdd + gnd
		# modify 12 if you change the circuit 
		self.layer_fc_2 = nn.Linear(n_latent_var + 4 * 12, n_latent_var)
		self.layer_fc_3 = nn.Linear(n_latent_var, out_dim)


	def forward(self, obs_gcn, obs_fc, adj):
		"""
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs_gcn, np.ndarray):
			obs_gcn = torch.tensor(obs_gcn, dtype=torch.float)

		if isinstance(obs_fc, np.ndarray):
			obs_fc = torch.tensor(obs_fc, dtype=torch.float)

		if isinstance(adj, np.ndarray):
			adj = torch.tensor(adj, dtype=torch.float)

		activation_gcn_1 = torch.tanh(self.layer_gcn_1(obs_gcn))
		activation_gcn_1 = torch.matmul(adj, activation_gcn_1)

		activation_gcn_2 = torch.tanh(self.layer_gcn_2(activation_gcn_1))
		activation_gcn_2 = torch.matmul(adj, activation_gcn_2)

		activation_gcn_3 = torch.tanh(self.layer_gcn_3(activation_gcn_2))
		activation_gcn_3 = torch.matmul(adj, activation_gcn_3)

		activation_fc_1 = torch.tanh(self.layer_fc_1(obs_fc))


		shape_dim = activation_gcn_3.dim()

		if shape_dim == 2:
			activation_gcn_3 = torch.flatten(activation_gcn_3)
			activation_fc_2 = torch.cat((activation_gcn_3, activation_fc_1), dim=-1)
			activation_fc_2 = torch.tanh(self.layer_fc_2(activation_fc_2))
			output = self.layer_fc_3(activation_fc_2)
		else:
			activation_gcn_3 = torch.flatten(activation_gcn_3, start_dim=1)
			activation_fc_2 = torch.cat((activation_gcn_3, activation_fc_1), dim=1)
			activation_fc_2 = torch.tanh(self.layer_fc_2(activation_fc_2))
			output = self.layer_fc_3(activation_fc_2)

		return output