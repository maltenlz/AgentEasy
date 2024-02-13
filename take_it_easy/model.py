import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
GAMMA = 0.9
LR = 1e-5
SIZE_SCALER = 4
TAU = 0.005
# Setup:
# predict value of the boardstate resulting from a move
# take the action with the highest value
# basically regression

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class EasyNet(nn.Module):
    """ DQN learning for Takeing It Easy!"""

    def __init__(self):
        super(EasyNet, self).__init__()
        # 19 tiles plus tile to be placed times 28 (all possible tiles plus no tile) one hot encoded remaining vector and remaining tiles
        self.layer1 = nn.Linear(20*28 + 28 + 1, 128*SIZE_SCALER)
        self.layer2 = nn.Linear(128*SIZE_SCALER, 128*SIZE_SCALER)
        self.layer3 = nn.Linear(128*SIZE_SCALER, 19)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)  # No activation function, raw Q-values
    
    def predict_value(self, x):
        state = torch.tensor(x, dtype = torch.float)
        return self(state).tolist()


class QTrainer:
    """ Class to perform training steps on the DQN """
    def __init__(self, policy_net, target_net, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.target_net = target_net
        self.policy_net = policy_net
        self.optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        # 1: predicted Q values with current state
        pred = self.policy_net(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + GAMMA * torch.max(self.target_net(next_state[idx]))
            target[idx][action[idx]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value=10.0)  # By value
        self.optimizer.step()
    
    def update_target(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """  
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)