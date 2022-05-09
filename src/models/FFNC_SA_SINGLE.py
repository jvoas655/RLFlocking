import sys
import os
from models.critic import *
import torch.nn.functional as F

__activations__ = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "tanh": torch.tanh
}

class FFNC_SA_SINGLE(Critic):
    def __init__(self, num_obs, num_acts, activation = "leaky_relu", hidden_size = 64):
        super().__init__(1, num_obs, num_acts)
        self.activation = activation
        self.FC1 = nn.Linear(num_obs+num_acts, hidden_size)
        self.FC2 = nn.Linear(hidden_size, hidden_size)
        self.FC3 = nn.Linear(hidden_size, hidden_size)
        self.FC4 = nn.Linear(hidden_size, 1)
    def forward(self, obs, acts=None):
        result = torch.cat((obs, acts), dim=1)
        result = __activations__[self.activation](self.FC1(result))
        result = __activations__[self.activation](self.FC2(result))
        result = __activations__[self.activation](self.FC3(result))
        result = self.FC4(result)
        return result
