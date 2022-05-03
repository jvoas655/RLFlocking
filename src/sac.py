import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from models.sac.model import GaussianPolicy, QNetwork, DeterministicPolicy, DiscretePolicy, QVFN, QVFNMinimal, QVFNKaiming
from utils.helpers import get_device
# from memory import ReplayMemory, Experience
from models.sac.replay_memory import ReplayMemory

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



class SAC(object):
    def __init__(self, n_agents, num_inputs, action_space, policy="Deterministic",
                    target_update_interval=50,
                    automatic_entropy_tuning=False,
                    gpuid=0,
                    hidden_size= 64,# 512,
                    lr=0.0003,
                    capacity=10000,
                    batch_size=256):
        self.n_agents = n_agents
        self.n_states = num_inputs
        self.n_actions = action_space.shape[0]
        self.gamma = 0.95 
        self.tau = 0.01 
        self.alpha = 0.1 
        self.episode_done = 0
        self.episodes_before_train = 10 
        self.steps_done = 0

        self.policy_type = policy 
        self.target_update_interval = target_update_interval 
        self.automatic_entropy_tuning = automatic_entropy_tuning 
        self.memory = ReplayMemory(capacity, 42)
        self.batch_size = batch_size
        self.device = get_device(gpuid)
        # print("self device", self.device)
        # self.device = "cpu" 

        # self.critic = QNetwork(n_agents, num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        # self.critic = QVFN(n_agents, num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic = QVFNMinimal(n_agents, num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        # self.critic = QVFNKaiming(n_agents, num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        # self.critic_target = QNetwork(n_agents, num_inputs, action_space.shape[0], hidden_size).to(self.device)
        # self.critic_target = QVFN(n_agents, num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_target = QVFNMinimal(n_agents, num_inputs, action_space.shape[0], hidden_size).to(self.device)
        # self.critic_target = QVFNKaiming(n_agents, num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        elif self.policy_type == "Deterministic":
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        
        
        elif self.policy_type == "Discrete":
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DiscretePolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state_batch, evaluate=False):
                # state_batch: n_agents x state_dim
        actions = torch.zeros(
            self.n_agents,
            self.n_actions).to(self.device)
        # print(state_batch.device)
        if not torch.is_tensor(state_batch):
            state_batch = torch.FloatTensor(state_batch)
        if state_batch.device != self.device:
            state_batch = state_batch.to(self.device)
        # state_batch = torch.FloatTensor(state_batch).to(self.device) 
        for i in range(self.n_agents):
            # state = torch.FloatTensor(state_batch[i, :]).to(self.device).unsqueeze(0)
            state = state_batch[i, :].unsqueeze(0)
            if evaluate is False:
                act, _, _ = self.policy.sample(state)
            else:
                _, _, act = self.policy.sample(state)

            
            mask = torch.zeros(actions.shape, device=actions.device, dtype=torch.bool)
            mask[i, :] = True
            actions = actions.masked_scatter(mask, act.clone())
        self.steps_done += 1
        return actions


        

    def update_policy(self):
        if self.episode_done <= self.episodes_before_train:
            return None, None


        for agent in range(self.n_agents):
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(2)
            mask_batch = torch.BoolTensor(mask_batch).to(self.device)

            # print("next_state_batch", next_state_batch, next_state_batch.shape)
            # print("mask batch", mask_batch, mask_batch.shape)
            non_final_next_states = next_state_batch[mask_batch]
            # print("non final next states", non_final_next_states)


            with torch.no_grad():
                # next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                non_final_next_action_log_pi = [self.policy.sample(non_final_next_states[:,i,:]) for i in range(self.n_agents)]
                
                non_final_next_action, non_final_next_log_pi, _ = zip(*non_final_next_action_log_pi)
                non_final_next_action = torch.stack(list(non_final_next_action)).transpose(0,1).to(self.device)
                if self.policy_type == "Gaussian":
                    non_final_next_log_pi = torch.stack(list(non_final_next_log_pi)).transpose(0,1).to(self.device)
                else:
                    non_final_next_log_pi = 0

                qf1_next_target, qf2_next_target = torch.zeros(self.batch_size, 1).to(self.device), torch.zeros(self.batch_size, 1).to(self.device)

                # qf1_next_target[mask_batch], qf2_next_target[mask_batch] = self.critic_target(non_final_next_states.view(-1, self.n_agents * self.n_states), non_final_next_action.contiguous().view(-1, self.n_agents * self.n_actions))
                qf1_next_target[mask_batch], qf2_next_target[mask_batch] = self.critic_target(non_final_next_states, non_final_next_action) # non final size * agent nums * observation size or action size

                # print("qf1 next target shape", qf1_next_target.shape)
                # print("qf2 next target shape", qf2_next_target.shape)
                # print("non final next log pi shape", non_final_next_log_pi.shape)
                next_state_log_pi = torch.zeros(self.batch_size, self.n_agents, 1).to(self.device)
                next_state_log_pi[mask_batch] = non_final_next_log_pi
                # print("next state log pi shape", next_state_log_pi.shape)
                # print("torch min shape", torch.min(qf1_next_target, qf2_next_target).shape)
                # print("next state log pi agent shape", next_state_log_pi[:, agent, :].shape)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi[:,agent,:]
                # print("min qf next target shape", min_qf_next_target.shape)
                # next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target) # already masked...
                # print("reward batch shape", reward_batch.shape)
                # print("reward batch agent shape", reward_batch[:,agent].shape)
                next_q_value = reward_batch[:,agent] + mask_batch.unsqueeze(1) * self.gamma * min_qf_next_target

            # qf1, qf2 = self.critic(state_batch.view(-1, self.n_agents * self.n_states), action_batch.contiguous().view(-1, self.n_agents * self.n_actions))  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            # print("qf1 shape", qf1.shape)
            # print("next q value", next_q_value.shape)
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            state_i = state_batch[:, agent, :]
            pi, log_pi, _ = self.policy.sample(state_i)
            
            ac = action_batch.clone() # batch_size * n_agents * action space dim
            # print("ac0", ac)
            m = torch.zeros(ac.shape, device=ac.device, dtype=torch.bool)
            m[:, agent, :] = True
            # print("m", m)
            ac = ac.masked_scatter(m, pi.clone()).detach()
            # print("ac shape", ac.shape)
            whole_action = ac.view(self.batch_size, -1)

            # qf1_pi, qf2_pi = self.critic(state_batch.view(-1, self.n_agents * self.n_states), whole_action)
            qf1_pi, qf2_pi = self.critic(state_batch, ac)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.steps_done % self.target_update_interval == 0 and self.steps_done > 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, checkpoint_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(checkpoint_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
    
    def to_float_tensor(self, data):
        # FloatTensor = torch.cuda.FloatTensor if self.device[:4] == "cuda" else torch.FloatTensor
        data = torch.FloatTensor(data).to(self.device)
        return data
