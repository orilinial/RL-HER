from collections import namedtuple
import random
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def moving_average(data_set, periods=10):
    weights = np.ones(periods) / periods
    averaged = np.convolve(data_set, weights, mode='valid')

    pre_conv = []
    for i in range(1, periods):
        pre_conv.append(np.mean(data_set[:i]))

    averaged = np.concatenate([pre_conv, averaged])
    return averaged
