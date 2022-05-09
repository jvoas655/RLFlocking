import numpy as np
import os
import torch
from models.discrete.value import QApproximationWithNN
from copy import deepcopy




class RandomPolicy:
    def __init__(self,
            num_agents:int,
            dim_obs:int,
            dim_act:int,

            gamma:float = 0.9,
            hidden:int = 32,
            # eps:float = 0.05,
            ):
        self.num_agents = num_agents
        self.dim_obs = dim_obs
        self.dim_act = dim_act


        self.steps_done = 0
        self.episode_done = 0
        
        norm = 0.1
        self.action_map = {0: np.array([0., 0.]),
                      1: np.array([0., 1.]) * norm,
                      2: np.sqrt(0.5) * (np.array([1., 1.])) * norm,
                      3: np.array([1., 0.]) * norm,
                      4: np.sqrt(0.5) * (np.array([1., -1.])) * norm,
                      5: np.array([0., -1.]) * norm,
                      6: np.sqrt(0.5) * (np.array([-1., -1.])) * norm,
                      7: np.array([-1., 0.]) * norm,
                      8: np.sqrt(0.5) * (np.array([-1., 1.])) * norm}

    def select_action(self, state):
        action = np.zeros((self.num_agents, self.dim_act))
        for agent in range(self.num_agents):
            act_idx = np.random.randint(len(self.action_map))
            action[agent] = self.action_map[act_idx] 
        self.steps_done += 1
        return action
