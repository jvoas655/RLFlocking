import sys
import os
from models.actor import *
import torch.nn.functional as F
import torch

__activations__ = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "tanh": torch.tanh
}

class FFNA(Actor):
    def __init__(self, num_obs, num_actions, activation = "leaky_relu"):
        super().__init__(num_obs, num_actions)
        self.activation = activation

        self.FC1 = nn.Linear(self.num_obs, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 64)
        self.FC4 = nn.Linear(64, self.num_actions)

    def forward(self, obs):
        #print("Actor")
        #print(obs)
        result = __activations__[self.activation](self.FC1(obs))
        #print(result)
        result = __activations__[self.activation](self.FC2(result))
        result = __activations__[self.activation](self.FC3(result))
        #print(result)
        result = self.FC4(result)
        result=F.normalize(result)
        #print(result)
        return result
