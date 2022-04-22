import sys
import os
from models.critic import *
import torch.nn.functional as F

__activations__ = {
    "relu": F.relu
}

class FFNC(Critic):
    def __init__(self, num_agents, num_obs, num_actions, activation = "relu"):
        super().__init__(num_agents, num_obs, num_actions)
        obs_dim = num_obs * num_agents
        act_dim = num_actions * num_agents
        self.activation = activation

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024+act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)
    def forward(self, obs, acts):
        x_obs = torch.Tensor(obs.flatten())
        x_acts = torch.Tensor(acts.flatten())
        x_fc1 = __activations__[self.activation](self.FC1(x_obs))
        x_comb = torch.cat((x_fc1, x_acts))
        x_fc2 = __activations__[self.activation](self.FC2(x_comb))
        x_fc3 = __activations__[self.activation](self.FC3(x_fc2))
        x_fc4 = self.FC4(x_fc3)
        return x_fc4
