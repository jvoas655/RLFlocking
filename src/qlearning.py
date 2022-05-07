import numpy as np
import os
import torch
from models.discrete.value import QApproximationWithNN
from copy import deepcopy




class Qlearning:
    def __init__(self,
            num_agents:int,
            dim_obs:int,
            dim_act:int,

            gamma:float = 0.9,
            # eps:float = 0.05,
            ):
        self.num_agents = num_agents
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.gamma = gamma
        # self.eps = eps

        self.QNet = QApproximationWithNN(num_agents=num_agents, state_dims=dim_obs, action_dims=dim_act, device="cpu")
        self.QNet_target = QApproximationWithNN(num_agents=num_agents, state_dims=dim_obs, action_dims=dim_act, device="cpu")
        # self.QNet_target = deepcopy(self.QNet)
        self.tau = 0.01

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

    def select_action(self, state, epsilon=0.05):
        action = np.zeros((self.num_agents, self.dim_act))
        for agent in range(self.num_agents):
            s = state[agent]
            Q = [self.QNet(s, self.action_map[a]) for a in self.action_map]
            if np.random.rand() < epsilon:
                act_idx = np.random.randint(len(self.action_map))
                action[agent] = self.action_map[act_idx]
            else:
                action[agent] = self.action_map[np.argmax(Q)]
        
        self.steps_done += 1
        return action

    def maxQ(self, s):
        maxq = -np.inf
        for a in self.action_map:
            qval = self.QNet_target(s, self.action_map[a])
            if qval > maxq:
                maxq = qval
        return maxq


    def update_policy(self, state, action, reward, next_state):
        # next_action = self.select_action(next_state, args.eps)
        for agent in range(self.num_agents):
            target = reward[agent] + self.gamma * self.maxQ(next_state[agent])
            self.QNet.update(state[agent], action[agent], target)
    
    def soft_update(self, target, source, t):
        for target_param, source_param in zip(target.model.parameters(), source.model.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
        
    def update_target(self):
        self.soft_update(self.QNet_target, self.QNet, self.tau)
    

    def save_checkpoint(self, checkpoint_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/{}_checkpoint_{}".format(checkpoint_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'qvalue_state_dict': self.QNet.model.state_dict(),
                    }, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.QNet.model.load_state_dict(checkpoint['qvalue_state_dict'])

