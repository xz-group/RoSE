import torch.nn as nn
import torch
from torch.distributions import Categorical
from model.gcn_fc_actor import GCN_FC_actor
from model.gcn_fc_critic import GCN_FC_critic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, in_dim_gcn, in_dim_spec, n_latent_var, out_dim):

        super(ActorCritic, self).__init__()
        # actor
        self.action_layer = GCN_FC_actor(in_dim_gcn, in_dim_spec, n_latent_var, out_dim)
        # critic
        self.value_layer = GCN_FC_critic(in_dim_gcn, in_dim_spec, n_latent_var, 1)


    def forward(self):
        raise NotImplementedError

    def act(self, state_gcn, state_spec, adj):


        state_gcn = torch.from_numpy(state_gcn).float().to(device)
        state_spec = torch.from_numpy(state_spec).float().to(device)
        action_probs = self.action_layer(state_gcn, state_spec, adj)

        dist = Categorical(action_probs)
        action = dist.sample()

        return action