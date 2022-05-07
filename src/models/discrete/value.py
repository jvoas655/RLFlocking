import torch
from torch import nn
import os
import numpy as np





class QValueNN(nn.Module):
    def __init__(self, num_agents, state_dims, action_dims, hidden_dim=32):
        super(QValueNN, self).__init__()

        self.num_agents = num_agents
        self.q = nn.Sequential(nn.Linear(state_dims + action_dims, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, 1))


    def forward(self, state, action): # state: batch_size * state_dims, action: batch_size * action_dims
        x = torch.cat([state, action], dim=1)
        # out = 0
        # for agent in range(self.num_agents):
        #     out += self.q(x[:,agent,:])

        # return out
        return self.q(x)


class QApproximationWithNN():
    def __init__(self,
                 num_agents,
                 state_dims,
                 action_dims,
                 device,
                 alpha=5e-5):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        self.device = device
        self.model = QValueNN(num_agents, state_dims, action_dims).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.loss_fn = nn.MSELoss()

    def __call__(self, s, a) -> float:
        # TODO: implement this method

        # raise NotImplementedError()
        if s.ndim == 1:
            s = np.expand_dims(s, 0)
        if a.ndim == 1:
            a = np.expand_dims(a, 0)

        self.model.eval()
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        q = self.model(s, a)
        return q.cpu().detach().numpy()

    def update(self, s, a, G):
        # TODO: implement this method
        # raise NotImplementedError()

        if s.ndim == 1:
            s = np.expand_dims(s, 0)
        if a.ndim == 1:
            a = np.expand_dims(a, 0)
        if G.ndim == 1:
            G = np.expand_dims(G, 0)

        self.model.train()
        self.optimizer.zero_grad()
        G = torch.FloatTensor(G).to(self.device)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        q = self.model(s, a)
        loss = 0.5 * self.loss_fn(q, G)
        loss.backward()
        self.optimizer.step()
    """
    def save_checkpoint(self, algo, checkpoint_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/{}_checkpoint_{}_{}".format(algo, checkpoint_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'qvalue_state_dict': self.model.state_dict(),
                    }, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['qvalue_state_dict'])
    """  
