import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from model.memory import Experience

SIZE_SCALER = 8

class EasyNet(nn.Module):
    """ DQN learning for Takeing It Easy!"""

    def __init__(self):
        super(EasyNet, self).__init__()
        # 19 tiles plus tile to be placed times 28 (all possible tiles plus no tile) one hot encoded remaining vector and remaining tiles
        self.layer1 = nn.Linear(20*28 + 27, 4*128*SIZE_SCALER)
        self.layer2 = nn.Linear(4 * 128*SIZE_SCALER, 128*SIZE_SCALER)
        self.layer3 = nn.Linear(128*SIZE_SCALER, 19)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)  # No activation function, raw Q-values
    
    def predict_value(self, x):
        state = torch.tensor(x, dtype = torch.float).to(device)
        preds = self(state)
        return preds.cpu().tolist()
    
    def save(self):
        torch.save(self, 'model_checkpoint.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class ThinkerConfig:
    ''' contains the parameters for the learning phase '''
    lr: float = 1e-5
    gamma: float = 0.99
    tau: float = 0.05
    batch_size: int = 128
    nsteps_target_update: int = 5

class Thinker:
    """ 
       Class that contains the thinking of the Agent. 
    """
    def __init__(
                 self,
                 nnet_class: type[nn.Module],
                 optimizer: optim.Optimizer = optim.Adam,
                 learning_config: ThinkerConfig = ThinkerConfig()
                 ):
        self.learning_config = learning_config
        self.target_net = nnet_class()
        self.policy_net = nnet_class()
        self.optimizer = optimizer(self.policy_net.parameters(), lr=learning_config.lr)
        self.criterion = nn.MSELoss()
        self.learning_steps = 0

    def learn_from_experience(
            self,
            replay_memory
            ):
        """
            Takes a Learning Step, based on the provided memory Sample
        """
        learning_batch = replay_memory.sample(self.learning_config.batch_size)
        state, next_state, action, reward, done, legal_moves = self._experiences_to_tensors(learning_batch)
        
        pred = self.policy_net(state)
        predicted_next_values = self.policy_net(next_state)

        # mask illegal next moves.
        masked_q_values = predicted_next_values.clone()
        masked_q_values[legal_moves == 0] = -float('inf')

        # from the legal moves see which will maximise the expected rewards
        next_actions = torch.argmax(masked_q_values, dim=1)
        future_Q_values = self.target_net(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # doesnt add future reward if it was the last move
        not_done_mask = torch.logical_not(done)  
        Q_new = reward + self.learning_config.gamma * future_Q_values * not_done_mask

        # Calculate TD errors with IS weights
        td_target =  pred[torch.arange(len(action)), action]

        # Calculate gradient-weighted loss 
        loss = self.criterion(td_target, Q_new.detach())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value = 3)
        self.optimizer.step()
        self.learning_steps += 1
        if self.learning_steps % self.learning_config.nsteps_target_update == 0:
            self._update_target_network()
    
    def predict(self, state):
        ''' returns predicted reward for all 19 Tiles (containing illegal moves) based on the current state.'''
        return self.policy_net.predict_value(state)
    
    def _update_target_network(self):
        """
            Soft update target model parameters. DDQN.
            θ_target = τ*θ_local + (1 - τ)*θ_target
        """  
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.learning_config.tau*policy_param.data + (1.0-self.learning_config.tau)*target_param.data)

    @staticmethod
    def _experiences_to_tensors(batch: list[Experience]) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """ Transforms the batch of EXperiences into torch tensors """
        state_t = torch.tensor(np.stack([e.state_t for e in batch]), dtype=torch.float).to(device)
        state_t1 = torch.tensor(np.stack([e.state_t1 for e in batch]), dtype=torch.float).to(device)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float).to(device)
        dones = torch.tensor([e.finished for e in batch], dtype=torch.bool).to(device)
        legal_moves = torch.tensor(np.stack([e.legal_moves for e in batch]), dtype=torch.long).to(device)
        return state_t, state_t1, actions, rewards, dones, legal_moves