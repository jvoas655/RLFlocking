import sys
import os
from models.actor import *
import torch.nn.functional as F
import torch

__activations__ = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu
}

class FFNA(Actor):
    def __init__(self, num_obs, num_actions, activation = "leaky_relu"):
        super().__init__(num_obs, num_actions)
        self.activation = activation

        self.FC1 = nn.Linear(self.num_obs, 32)
        self.FC2 = nn.Linear(32, 32)
        self.FC3 = nn.Linear(32, 32)
        self.FC4 = nn.Linear(32, self.num_actions)

    def forward(self, obs):
        result = __activations__[self.activation](self.FC1(obs))
        result = __activations__[self.activation](self.FC2(result))
        result = __activations__[self.activation](self.FC3(result))
        result = self.FC4(result)
        return result
