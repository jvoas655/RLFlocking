import sys
import os
from models.actor import *
import torch.nn.functional as F
import torch

__activations__ = {
    "relu": F.relu
}

class FFNA(Actor):
    def __init__(self, num_obs, num_actions, activation = "relu"):
        super().__init__(num_obs, num_actions)
        self.activation = activation

        self.FC1 = nn.Linear(self.num_obs, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, self.num_actions + 1)

    # action output between -2 and 2
    def forward(self, obs):
        result = __activations__[self.activation](self.FC1(obs))
        result = __activations__[self.activation](self.FC2(result))
        result = self.FC3(result)
        result = F.normalize(result[:, :self.num_actions]) * torch.abs(torch.clamp(result[:, self.num_actions], min = -1.0, max = 1.0))
        return result
