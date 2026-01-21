import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torchtyping import TensorType


@dataclass
class Experience:
    """
    container class to store experiences, to be learned from later
    """

    state_t: TensorType[torch.float32]
    state_t1: TensorType[torch.float32]
    action: TensorType[torch.int64]
    reward: TensorType[torch.float32]
    finished: TensorType[torch.bool]
    legal_moves: TensorType[torch.long]


class ABCMemory(ABC):
    """Object that stores and retrieves batches of Experiences from exploration."""

    @abstractmethod
    def add(self, new_experience: Experience) -> None:
        """commits a new Experience to memory"""
        
    @abstractmethod
    def sample(
        self, batch_size: int
    ) -> tuple[list[int], list[float], list[Experience]]:
        """Sample from the Replay buffer
        Args:
            int: batch size
        Returns:
            tuple conising of
            Optional[list]: list of indices in buffer, used in REP
            Optional[list]: list of priority values
            list: the Experiences sampled to be learned from
        """

    @abstractmethod
    def update(self, idx: int, error: float):
        """updates the priority for future sampling"""


class SimpleReplayBuffer(ABCMemory):
    """Simple Replay Buffer (Uniform Sampling)"""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, new_experience: Experience):
        self.memory.append(new_experience)

    def sample(self, batch_size) -> tuple[int | None, list[float], list[Experience]]:
        batch = random.sample(self.memory, batch_size)
        priority = [1] * batch_size
        pseudo_idx = [1] * batch_size
        return pseudo_idx, priority, batch

    def update(self, idx: int, error: float):
        """uniform sampling, so no updates required."""

    def __len__(self):
        return len(self.memory)


class PERBuffer(ABCMemory):
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5
        self.beta = 0.4
        self.beta_annealing_rate = 1.002

    def add(self, experience: Experience):
        leaf_nodes = self.tree.tree[
            -self.tree.capacity : -self.tree.capacity + self.tree.size
        ]
        if len(leaf_nodes) == 0:
            max_p = 1.0
        else:
            max_p = np.max(leaf_nodes)
            if max_p == 0:
                max_p = 1.0

        self.tree.add(max_p, experience)

    def sample(self, n):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total / n
        self.beta = min(self.beta * 1.002, 1.0)
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        total = self.tree.total
        probs = np.array(priorities) / total
        weights = (self.tree.size * probs + 1e-8) ** -self.beta
        weights /= weights.mean()
        return idxs, weights, batch

    def update(self, idx, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.update(idx, p)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.write = 0

    def add(self, priority, data):
        assert priority >= 0, "Trying to add experience with negative priority."
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        assert priority >= 0, "Trying to update priority to negative value."
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]
