import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        self.num_obs = num_obs
        self.num_actions = num_actions
    def forward(self):
        raise NotImplementedError
