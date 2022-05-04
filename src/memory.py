from collections import namedtuple
import random
Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        #print(self.memory[self.position])
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    def clear(self):
        self.memory = []
        self.position = 0

class PAReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.sp = 0

    def push(self, n_agents, state, action, next_state, rewards):
        agent_ind = 0
        while len(self.memory) < self.capacity and agent_ind < n_agents:
            self.memory.append(Experience(state[agent_ind, :], action[agent_ind], next_state[agent_ind, :], rewards[agent_ind]))
            self.position = (self.position + 1) % self.capacity
            agent_ind += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    def clear(self):
        self.memory = []
        self.position = 0
