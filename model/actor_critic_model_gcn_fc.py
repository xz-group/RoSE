import torch.nn as nn
import torch
from torch.distributions import Categorical
from model.gcn_fc_actor import GCN_FC_actor
from model.gcn_fc_critic import GCN_FC_critic
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_gcn_dim, state_spec_dim, n_latent_var, action_dim):

        super(ActorCritic, self).__init__()
        # actor
        self.action_layer = GCN_FC_actor(state_gcn_dim, state_spec_dim, n_latent_var, action_dim)
        # critic
        self.value_layer = GCN_FC_critic(state_gcn_dim, state_spec_dim, n_latent_var, 1)


    def forward(self):
        raise NotImplementedError

    def act(self, state_gcn, state_spec, adj, memory):

        state_gcn = torch.from_numpy(state_gcn).float().to(device)
        state_spec = torch.from_numpy(state_spec).float().to(device)
        action_probs = self.action_layer(state_gcn, state_spec, adj)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob_sum = torch.sum(log_prob, dim=0)

        memory.states_gcn.append(state_gcn)
        memory.states_spec.append(state_spec)
        memory.actions.append(action)
        memory.logprobs.append(log_prob_sum)

        return action.detach()

    def evaluate(self, state_gcn, state_spec, adj, action):

        action_probs = self.action_layer(state_gcn, state_spec, adj)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        log_prob_sum = torch.sum(action_logprobs, dim=1)
        dist_entropy = dist.entropy()
        dist_entropy_sum = torch.sum(dist_entropy, dim=1)

        state_value = self.value_layer(state_gcn, state_spec, adj)

        return log_prob_sum, torch.squeeze(state_value), dist_entropy_sum
