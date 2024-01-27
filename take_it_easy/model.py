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
GAMMA = 0.95
LR = 1e-5
SIZE_SCALER = 16
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
    """ DDQN learning for Takeing It Easy!"""

    def __init__(self):
        super(EasyNet, self).__init__()
        self.layer1 = nn.Linear(20*10, 128*SIZE_SCALER)
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
    def __init__(self, policy_net, target_net, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        #self.target_net = target_net
        self.policy_net = policy_net
        self.optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        self.criterion = nn.SmoothL1Loss()

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
                Q_new = reward[idx] + GAMMA * torch.max(self.policy_net(next_state[idx]))
            target[idx][action[idx]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
    
    # def update_target(self):
    #     self.target_net.load_state_dict(self.policy_net.state_dict())