from models.FFNC_SINGLE import *
from models.FFNA import *
import torch
from copy import deepcopy
from memory import PAReplayMemory
from torch.optim import Adam
import torch.nn as nn
import numpy as np


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class PS_CACER:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actor = FFNA(dim_obs, dim_act)
        self.critic = FFNC_SINGLE(dim_obs)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = PAReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = 0.1
        self.critic_optimizer = Adam(self.critic.parameters(),lr=0.01)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=0.001)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()

        self.episode_done = 0
        #torch.autograd.set_detect_anomaly(True)

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        transitions = self.memory.sample(self.batch_size)
        temporal_buffer = []
        deltas = []
        for transition_ind in range(len(transitions)):
            deltas.append(transitions[transition_ind].rewards + self.GAMMA * self.critic(FloatTensor(transitions[transition_ind].next_states)) - self.critic(FloatTensor(transitions[transition_ind].states)))
            if (deltas[transition_ind] > 0):
                temporal_buffer.append(transitions[transition_ind])
        deltas = torch.stack(deltas)
        if (len(temporal_buffer)):
            temporal_actions = []
            temporal_states = []
            for buffer_ind in range(len(temporal_buffer)):
                temporal_actions.append(temporal_buffer[buffer_ind].actions.clone())
                temporal_states.append(FloatTensor(temporal_buffer[buffer_ind].states))
            temporal_actions = torch.stack(temporal_actions)
            temporal_states = torch.stack(temporal_states)
            self.actor_optimizer.zero_grad()
            actor_loss = torch.div(torch.sum(torch.norm(torch.sub(temporal_actions.detach(), self.actor(temporal_states)), dim = 1)), len(temporal_buffer)).clone()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss = torch.div(torch.sum(torch.pow(torch.norm(deltas, dim=1), 2)), self.batch_size)
            critic_loss.backward()
            self.critic_optimizer.step()


            #print(actor_loss)
        #print(len(temporal_buffer), "/", self.batch_size)
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
            act = self.actor(sb.unsqueeze(0).clone()).squeeze()
            if (noise):
                act = act + torch.from_numpy((np.random.randn(2) - 0.5) * self.var).type(FloatTensor)
                act_norm = torch.clamp(torch.linalg.norm(act), min=1.0)
                act = torch.div(act, act_norm)
            actions.append(act)
        if noise and self.episode_done > self.episodes_before_train and self.var > 0.005:
            self.var *= 0.999998
        return actions
