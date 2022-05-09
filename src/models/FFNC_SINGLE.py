import sys
import os
from models.critic import *
import torch.nn.functional as F

__activations__ = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "tanh": torch.tanh
}

class FFNC_SINGLE(Critic):
    def __init__(self, num_obs, activation = "leaky_relu"):
        super().__init__(1, num_obs, 0)
        self.activation = activation
        self.FC1 = nn.Linear(self.num_obs, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, 64)
        self.FC4 = nn.Linear(64, 1)
    def forward(self, obs, acts=None):
        #print("Critic")
        #print(obs)
        x_fc1 = __activations__[self.activation](self.FC1(obs))
        #print(x_fc1)
        #x_comb = torch.cat((x_fc1, x_acts), dim=1)
        x_fc2 = __activations__[self.activation](self.FC2(x_fc1))
        #print(x_fc2)
        x_fc3 = __activations__[self.activation](self.FC3(x_fc2))
        x_fc4 = self.FC4(x_fc3)
        #print(x_fc3)
        return x_fc4
