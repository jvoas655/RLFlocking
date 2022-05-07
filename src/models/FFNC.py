import sys
import os
from models.critic import *
import torch.nn.functional as F

__activations__ = {
    "relu": F.relu,
    "tanh": torch.tanh
}

class FFNC(Critic):
    def __init__(self, num_agents, num_obs, num_actions, activation = "relu"):
        super().__init__(num_agents, num_obs, num_actions)
        obs_dim = num_obs * num_agents
        act_dim = num_actions * num_agents
        self.activation = activation
        print(act_dim, obs_dim)
        self.FC1 = nn.Linear(obs_dim, 256)
        self.BN1 = nn.BatchNorm1d(256)
        self.FC2 = nn.Linear(256+act_dim, 256)
        self.BN2 = nn.BatchNorm1d(256)
        self.FC3 = nn.Linear(256, 32)
        self.BN3 = nn.BatchNorm1d(32)
        self.FC4 = nn.Linear(32, 1)
    def forward(self, obs, acts):
        x_obs = obs
        x_acts = acts
        x_fc1 = __activations__[self.activation](self.BN1(self.FC1(x_obs)))

        x_comb = torch.cat((x_fc1, x_acts), dim=1)
        x_fc2 = __activations__[self.activation](self.BN2(self.FC2(x_comb)))
        x_fc3 = __activations__[self.activation](self.BN3(self.FC3(x_fc2)))
        x_fc4 = self.FC4(x_fc3)
        return x_fc4
