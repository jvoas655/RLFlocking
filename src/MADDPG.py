from models.FFNC import *
from models.FFNA import *
import torch
from copy import deepcopy
from memory import ReplayMemory, Experience
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


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [FFNA(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [FFNC(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [0.1 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.1) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.01) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
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
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = BoolTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            state_batch = list(map(lambda s: FloatTensor(s), batch.states))
            action_batch = batch.actions
            reward_batch = list(map(lambda s: FloatTensor(s), batch.rewards))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(state_batch)
            action_batch = torch.stack(action_batch)
            reward_batch = torch.stack(reward_batch)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = torch.stack(
                [FloatTensor(s) for s in batch.next_states
                 if s is not None])

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1).detach()
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = torch.zeros(
                self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * 0.01)
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()


            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            mask = torch.zeros(ac.shape, device=ac.device, dtype=torch.bool)
            mask[:, agent, :] = True
            ac = ac.masked_scatter(mask, action_i.clone()).detach()
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()

            actor_loss.backward()

            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss
    def to_float_tensor(self, data):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        data = FloatTensor(data)
        return data

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        actions = torch.zeros(
            self.n_agents,
            self.n_actions).type(FloatTensor)
        state_batch = FloatTensor(state_batch)
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            #act = act + torch.from_numpy(
                #(np.random.randn(2)) * self.var[i]).type(FloatTensor)
            #act_norm = torch.clamp(torch.linalg.norm(act), min=1.0)
            #act = torch.div(act, act_norm)
            if self.episode_done > self.episodes_before_train and\
               self.var[i] > 0.005:
                self.var[i] *= 0.999998
            mask = torch.zeros(actions.shape, device=actions.device, dtype=torch.bool)
            mask[i, :] = True
            actions = actions.masked_scatter(mask, act.clone())
        self.steps_done += 1
        return actions
