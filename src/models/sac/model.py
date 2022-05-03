import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def kaiming_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_agents, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        obs_dim = num_agents * num_inputs
        act_dim = num_agents * num_actions

        # Q1 architecture
        self.linear1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                # (action_space.high - action_space.low) / 2.)
                (action_space[0] - action_space[1]) / 2.)
            self.action_bias = torch.FloatTensor(
                # (action_space.high + action_space.low) / 2.)
                (action_space[0] + action_space[1]) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                # (action_space.high - action_space.low) / 2.)
                (action_space[0] - action_space[1]) / 2.)
            self.action_bias = torch.FloatTensor(
                # (action_space.high + action_space.low) / 2.)
                (action_space[0] + action_space[1]) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class DiscretePolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DiscretePolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # self.mean = nn.Linear(hidden_dim, num_actions)
        self.mean = nn.Linear(hidden_dim, 9)
        # self.noise = torch.Tensor(num_actions)
        self.m = nn.Softmax(dim=1)


        self.apply(weights_init_)

        norm = 0.1
        embedding_weights = [torch.FloatTensor([0, 0]),
                             torch.FloatTensor([0, 1]),
                             torch.sqrt(torch.tensor(0.5)) * (torch.FloatTensor([1, 1])), 
                             torch.FloatTensor([1, 0]),
                             torch.sqrt(torch.tensor(0.5)) * (torch.FloatTensor([1, -1])), 
                             torch.FloatTensor([0, -1]),
                             torch.sqrt(torch.tensor(0.5)) * (torch.FloatTensor([-1, -1])), 
                             torch.FloatTensor([-1, 0]),
                             torch.sqrt(torch.tensor(0.5)) * (torch.FloatTensor([-1, 1]))]

        embedding_weights = torch.stack(embedding_weights) * norm
        self.embedding = nn.Embedding(9, 2)
        self.embedding.weight = torch.nn.Parameter(embedding_weights)
        self.embedding.weight.requires_grad = False 
        



    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.m(self.mean(x))

    def sample(self, state):
        prob = self.forward(state)
        action = torch.multinomial(prob, num_samples=1).squeeze(1)
        maxact = torch.argmax(prob, dim=1)
        # print("prob", prob)
        # print("act", action, action.shape)
        # print("maxact", maxact, maxact.shape)
        return self.embedding(action), torch.tensor(0.), self.embedding(maxact)

    def to(self, device):
        return super(DiscretePolicy, self).to(device)

"""
policy = DiscretePolicy(45, 2, 64)
state = torch.rand(2, 45)
act, _, maxact = policy.sample(state)
print("act embedding", act, act.shape)
print("max act embedding", maxact, maxact.shape)
"""




class QVFN(nn.Module):
    def __init__(self, num_agents, num_inputs, num_actions, hidden_dim):
        super(QVFN, self).__init__()
        
        self.num_agents = num_agents
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        # Q1 architecture
        # self.q1_list = nn.ModuleList([nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, 1)) for _ in range(num_agents)])
        
        self.q1 = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, 1))

        # Q2 architecture
        # self.q2_list = nn.ModuleList([nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, 1)) for _ in range(num_agents)])
        self.q2 = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, 1))

        self.apply(weights_init_)

    def forward(self, state, action): # don't flatten state: num_agents * observation space (45), action: num_agents * action space (2)
        xu = torch.cat([state, action], dim=2)
       
        x1 = 0
        for agent in range(self.num_agents):
            x1 += self.q1(xu[:,agent,:])

        x2 = 0
        for agent in range(self.num_agents):
            # xb = xu[:,agent,:]
            # xb = self.q2_list[agent](xb)
            x2 += self.q2(xu[:,agent,:])

        return x1, x2


class QVFNMinimal(nn.Module):
    def __init__(self, num_agents, num_inputs, num_actions, hidden_dim=64):
        super(QVFNMinimal, self).__init__()
        
        self.num_agents = num_agents
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        # Q1 architecture
        # self.q1_list = nn.ModuleList([nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, 1)) for _ in range(num_agents)])
        
        self.q1 = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, 1))

        # Q2 architecture
        # self.q2_list = nn.ModuleList([nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, 1)) for _ in range(num_agents)])
        self.q2 = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, 1))

        self.apply(weights_init_)

    def forward(self, state, action): # don't flatten state: num_agents * observation space (45), action: num_agents * action space (2)
        xu = torch.cat([state, action], dim=2)
       
        x1 = 0
        for agent in range(self.num_agents):
            x1 += self.q1(xu[:,agent,:])

        x2 = 0
        for agent in range(self.num_agents):
            # xb = xu[:,agent,:]
            # xb = self.q2_list[agent](xb)
            x2 += self.q2(xu[:,agent,:])

        return x1, x2


class QVFNKaiming(nn.Module):
    def __init__(self, num_agents, num_inputs, num_actions, hidden_dim):
        super(QVFNKaiming, self).__init__()
        
        self.num_agents = num_agents
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        # Q1 architecture
        # self.q1_list = nn.ModuleList([nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, 1)) for _ in range(num_agents)])
        
        self.q1 = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, 1))

        # Q2 architecture
        # self.q2_list = nn.ModuleList([nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, hidden_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(hidden_dim, 1)) for _ in range(num_agents)])
        self.q2 = nn.Sequential(nn.Linear(num_inputs + num_actions, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, 1))

        self.apply(kaiming_init_)

    def forward(self, state, action): # don't flatten state: num_agents * observation space (45), action: num_agents * action space (2)
        xu = torch.cat([state, action], dim=2)
       
        x1 = 0
        for agent in range(self.num_agents):
            x1 += self.q1(xu[:,agent,:])

        x2 = 0
        for agent in range(self.num_agents):
            # xb = xu[:,agent,:]
            # xb = self.q2_list[agent](xb)
            x2 += self.q2(xu[:,agent,:])

        return x1, x2
