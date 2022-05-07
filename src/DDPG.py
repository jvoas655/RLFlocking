from models.FFNC_SA_SINGLE import *
from models.FFNA import *
import torch
from copy import deepcopy
from memory import PAReplayMemory
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import random


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class DDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = []
        self.actor_targets = []
        for i in range(n_agents):
            self.actors.append(FFNA(dim_obs, dim_act))
            self.actor_targets.append(deepcopy(self.actors[-1]))
        self.critic = FFNC_SA_SINGLE(dim_obs, dim_act)
        self.critic_target = deepcopy(self.critic)
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = PAReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = 1.0
        self.critic_optimizer = Adam(self.critic.parameters(),lr=0.0001)
        self.actor_optimizers = []
        for i in range(n_agents):
            self.actor_optimizers.append(Adam(self.actors[i].parameters(), lr=0.0001))

        self.critic_criterion = nn.MSELoss()

        if self.use_cuda:
            for i in range(n_agents):
                self.actors[i].cuda()
                self.actor_targets[i].cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        self.episode_done = 0
        #torch.autograd.set_detect_anomaly(True)

    def update_policy(self, exp = None):
        # do not train until exploration is enough
        if self.episode_done < self.episodes_before_train:
            return 0, 0

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        if (exp == None):
            transitions = self.memory.sample(self.batch_size)
        else:
            transitions = random.sample(exp, self.batch_size)
        rewards = FloatTensor(np.array(list(map(lambda t: t.rewards, transitions)))).view(self.batch_size, -1)
        states = FloatTensor(np.array(list(map(lambda t: t.states, transitions)))).view(self.batch_size, -1)
        next_states = FloatTensor(np.array(list(map(lambda t: t.next_states, transitions)))).view(self.batch_size, -1)
        actions = torch.stack(list(map(lambda t: t.actions.clone(), transitions))).view(self.batch_size, -1).detach()

        values = self.critic(states, actions)
        next_actions = self.actor_targets[0](next_states).detach()
        next_values = rewards + self.GAMMA * self.critic_target(next_states, next_actions)

        critic_loss = self.critic_criterion(next_values, values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actors[0](states)).mean()
        self.actor_optimizers[0].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[0].step()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_targets[0], self.actors[0], self.tau)

        return critic_loss.item(), actor_loss.item()
    def to_float_tensor(self, data):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        data = FloatTensor(data)
        return data

    def select_action(self, state_batch, noise=True):
        # state_batch: n_agents x state_dim
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        actions = []
        state_batch = FloatTensor(state_batch)
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[0](sb.unsqueeze(0).clone()).squeeze()
            if (noise):
                act = act + torch.from_numpy(np.random.randn(2) * self.var).type(FloatTensor)
                act = nn.functional.normalize(act.view(1, -1))
            actions.append(act)
        if noise and self.episode_done > self.episodes_before_train and self.var > 0.005:
            self.var *= 0.999998
        return actions
