import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, num_agents, num_obs, num_actions):
        super().__init__()
        self.num_agents = num_agents
        self.num_obs = num_obs
        self.num_actions = num_actions
    def forward(self):
        raise NotImplementedError
