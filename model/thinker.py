import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from model.memory import Experience
from torchview import draw_graph

class EasyNet(nn.Module):
    """ DQN learning for Takeing It Easy!"""

    def __init__(self, size_scaler, input_dim):
        super(EasyNet, self).__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim, 128*size_scaler)
        self.layer2 = nn.Linear(128*size_scaler, 128*size_scaler)
        self.layer3 = nn.Linear(128*size_scaler, 19)
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
    
    def plot_nnet(self):
        input_size = (1, self.input_dim)  # batch size 1
        model_graph = draw_graph(self, input_size=input_size, graph_name='EasyNet', save_graph=True)
        model_graph.visual_graph.render("nnet_architecture", format="png")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class LearningConfig:
    ''' contains the parameters for the learning phase '''
    lr: float = 1e-5
    gamma: float = 0.99
    tau: float = 0.05
    batch_size: int = 128
    nsteps_target_update: int = 5
    size_scaler: int = 4
    weight_decay: float = 1e-4

class Thinker:
    """ 
       Class that contains the thinking of the Agent. 
    """
    def __init__(
                 self,
                 learning_config: LearningConfig = LearningConfig()
                 ):
        self.learning_config = learning_config
        self.target_net = None
        self.policy_net = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.learning_steps = 0
    
    def initialize_nnets(self, input_dims):
        self.target_net = EasyNet(size_scaler=LearningConfig.size_scaler, input_dim=input_dims)
        self.policy_net = EasyNet(size_scaler=LearningConfig.size_scaler, input_dim=input_dims)
        self.optimizer = optim.AdamW(params=self.policy_net.parameters(), lr=self.learning_config.lr, weight_decay=self.learning_config.weight_decay)

    def learn_from_experience(
            self,
            replay_memory
            ):
        """
            Takes a Learning Step, based on the provided memory Sample
        """
        idxs, weights, learning_batch = replay_memory.sample(self.learning_config.batch_size)
        state, next_state, action, reward, done, legal_moves = self._experiences_to_tensors(learning_batch)
        weights = torch.tensor(weights, dtype=torch.float32, device=state.device)

        pred = self.policy_net(state)
        predicted_next_values = self.policy_net(next_state)

        # mask illegal next moves.
        masked_q_values = predicted_next_values.clone()
        masked_q_values[legal_moves == 0] = -float('inf')

        # from the legal moves see which will maximise the expected rewards
        next_actions = torch.argmax(masked_q_values, dim=1)
        future_q_values = self.target_net(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # doesnt add future reward if it was the last move
        not_done_mask = torch.logical_not(done)  
        q_new = reward + self.learning_config.gamma * future_q_values * not_done_mask

        # Calculate TD errors with IS weights
        td_target =  pred[torch.arange(len(action)), action]
        elementwise_loss = self.criterion(td_target, q_new.detach())
        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value = 3)
        self.optimizer.step()
        self.learning_steps += 1

        if replay_memory.memory_type == 'PER':
            td_errors = (q_new - td_target).detach().cpu().numpy()
            for idx, error in zip(idxs, td_errors):
                replay_memory.update(idx, error)

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