from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
import random

@dataclass
class Experience:
    ''' 
        container class to store experiences, to be learned from later
    '''
    state_t: np.array
    state_t1: np.array
    action: int
    reward: float
    finished : bool
    legal_moves: np.array

class Memory(ABC):
    ''' Object that stores and retrieves batches of Experiences from exploration.'''
    @abstractmethod
    def add(self, new_experience: Experience) -> None:
        ''' commits a new Experience to memory'''
        
    @abstractmethod
    def sample(self, batch_size: int) -> list[Experience]:
        ''' returns a random sample of batch_size Experiences '''

class SimpleReplayBuffer(Memory):
    """ Sum Tree for Prioritized Experience Replay """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, new_experience: Experience):
        self.memory.append(new_experience)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch


class PrioritizedReplayMemory:
    """ Sum Tree for Prioritized Experience Replay """
    def __init__(self, capacity):
        self.capacity = capacity 
        self.tree = np.zeros(2 * capacity - 1)  # Tree nodes (double the capacity) 
        self.data = np.zeros(capacity, dtype=object)  # Store experiences
        self.write_index = 0  # Track where to write next experience
        self.max_entries = capacity - 1  # Max number of entries
        self.max_priority = 1.0  # Max priority
        self.beta_0 = 0.4  # Importance sampling exponent
        self.beta = self.beta_0  
        self.max_it = 1600000  # Max iterations
        self.iteration = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # Calculate parent index
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1  # Left child index
        right = left + 1  # Right child index

        if left >= len(self.tree):  # Reached leaf node
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]  # Total priority stored in the root node

    def add(self, data):
        idx = self.write_index + self.max_entries

        self.data[self.write_index] = data  # Store data 
        self.update(idx, self.max_priority)  # For newly added use the max priority

        self.write_index += 1
        if self.write_index >= self.capacity:  # Reset write index 
            self.write_index = 0

    def update_batch(self, idxs, priorities):
        ### Update after learning step ###
        for idx, p in zip(idxs, priorities):
            self.update(idx, p)
        ### also update the max priority ###
        new_max = max(priorities)
        self.max_priority = max(new_max, self.max_priority)

        ### and the beta value ###
        self.iteration += 1
        beta = self.beta_0 + (1 - self.beta_0) * (self.iteration / self.max_it)
        self.beta =  beta

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)  # Update values to root

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.max_entries
        return idx, self.tree[idx], self.data[data_idx]
    
    def sample(self, batch_size):
        """ Sample from this sumtree """
        batch_idx = []
        batch_priorities = []
        segment = self.total() / batch_size  
        data_ls = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)

            idx, p, data = self.get(s)
            batch_idx.append(idx)
            batch_priorities.append(p)
            data_ls.append(data)
        return batch_idx, batch_priorities, data_ls

    def get_is_weights(self, idxs):
        # Sample probabilities are stored as priorities in the SumTree
        probabilities = [self.get(idx)[1] for idx in idxs] 
        N = len(self.data)  # Replay buffer size
        max_weight = (N * min(probabilities) + 0.0001) ** (-self.beta) 

        weights = [(N * p + 0.0001) ** (-self.beta) / max_weight for p in probabilities]  
        return weights 