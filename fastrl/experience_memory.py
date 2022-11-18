
from collections import deque, namedtuple
import random
import numpy as np
import torch

class ExperienceMemory:
    """Simle experience replay memory for reinforcement algorithms."""

    _fields = ("state", "action", "reward", "next_state", "done")
    
    def __init__(self, buffer_size=1024, device="cpu"):
        self._buffer_size = buffer_size
        self._data_size = 0
        self._device = device
        self._memory = {}
        self._memory["state"] = deque(maxlen=buffer_size) 
        self._memory["action"] = deque(maxlen=buffer_size)
        self._memory["reward"] = deque(maxlen=buffer_size)
        self._memory["next_state"] = deque(maxlen=buffer_size)
        self._memory["done"] = deque(maxlen=buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        if self._data_size < self._buffer_size:
            self._data_size += 1 
        self._memory["state"].append(state)
        self._memory["action"].append(action)
        self._memory["reward"].append(reward)
        self._memory["next_state"].append(next_state)
        self._memory["done"].append(done)

    def _sample_by_indices(self, indices, fields):
        if fields is None:
            fields = ExperienceMemory._fields
        output = []
        for field in fields:
            sampled = [self._memory[field][i] for i in indices]
            stacked = torch.from_numpy(np.stack(sampled, axis=0)).float().to(self._device)
            output.append(stacked)
        return tuple(output)

    def sample_random(self, sample_size, fields=None):
        indices = random.sample(range(self._data_size), sample_size)
        return self._sample_by_indices(indices, fields)

    def sample_sequential(self, sample_size, fields=None):
        last_index = random.randint(sample_size, self._data_size) - 1
        indices = list(range(last_index, last_index-sample_size, -1))
        return self._sample_by_indices(indices, fields)

    def sample_all(self, fields=None):
        indices = list(range(self._data_size))
        return self._sample_by_indices(indices, fields)

    def sample_last_k(self, k, fields=None):
        if k > self._data_size:
            k = self._data_size
        indices = list(range(-1, -k-1, -1))
        return self._sample_by_indices(indices, fields)
        
    def __len__(self):
        return len(self._data_size)

    @property
    def size(self):
        return self._data_size
    

if __name__=="__main__":
    em = ExperienceMemory(100)
    for i in range(8):
        state = np.random.randn(10)
        action = np.random.rand(3)
        reward = np.random.rand(1)
        next_state = np.random.randn(10)
        done = np.random.rand(1)>0.5
        em.add(state, action, reward, next_state, done)