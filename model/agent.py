from take_it_easy.board import Board
import random
from model.thinker import Thinker
from take_it_easy.value_functions import actual_score
from model.memory import  Memory, Experience
import math
import numpy as np
import json
import torch
import itertools

##### Only used if exploration strateguy is epsilon greedy
EPS_START = 0.5
EPS_END = 0.025
EPS_DECAY = 150000
MAX_MEMORY = 20000

##### For Boltzman exploration
TAU_MAX = 1
#### decay_rate = -ln(0.01 / 1.0) / 300000  ≈ 0.0000154 expecting we are decent after 300k games
DECAY_RATE = -math.log(0.01 / 1.0) / 1000000
print(f'Boltzman decay rate: {DECAY_RATE}')

def number_to_str(numbers: tuple[int]):
    return ''.join(str(n) for n in numbers)

ALL_NUMBERS = list(itertools.product([1,5,9], [3,4,8], [2,6,7]))
TILE_MAPPING = {number_to_str(n): i+1 for i, n in enumerate(ALL_NUMBERS)}
TILE_MAPPING[number_to_str((0,0,0))] = 0
print(TILE_MAPPING)

class AgentEasy:
    """ Agent that plays and learns the game."""

    def __init__(
            self,
            thinker: Thinker,
            memory: Memory
            ):
        self.thinker = thinker
        self.replay_memory = memory
        self.exploration_strategy = choose_epsilon_greedy
        self.value_function = actual_score
        self.eps_start = 0.25
        self.eps_end = 0.05
        self.eps_decay = 50000
        self.n_actions = 0

    def translate_board(self, board: Board):
        """ Returns the state of the board """
        x_input = []
        for row in board.tiles:
            for tile in row:
                x_input += one_hot_encode(tile)
        x_input += one_hot_encode(board.current_tile)
        return x_input
    
    def act_and_observe_action(
            self,
            board: Board
            ):
        state_t = self.translate_board(board)
        score_t = self.value_function(board)
        loc, predicted_action = self.exploration_strategy(
                                                          self, 
                                                          state_t, 
                                                          board.get_open_moves(),
                                                          self.n_actions
                                                          )
        board.action_by_id(loc)
        state_t1 = self.translate_board(board)
        score_t1 = self.value_function(board)
        reward = score_t1 - score_t
        legal_moves = board.get_open_moves()
        idx_legal_move = [d[1] for d in legal_moves]
        legal_moves_mask = [1 if i in idx_legal_move else 0 for i in range(0, 19)]
        new_experience = Experience(state_t, state_t1, predicted_action, reward, board.finished, legal_moves_mask)
        self.replay_memory.add(new_experience)
        self.n_actions += 1

    def load_nnet(self):
        self.thinker.policy_net = torch.load('model_checkpoint.pth')

    def learn(self):
        self.thinker.learn_from_experience(self.replay_memory)

    def save_nnet(self):
        self.thinker.policy_net.save()



def choose_epsilon_greedy(agent, state, possible_moves, current_episode = 0):
    """ Chooses the next move based on the epsilon greedy strategy """
    if possible_moves:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * current_episode / EPS_DECAY)
        if sample > eps_threshold:
            idxs = [move[1] for move in possible_moves]
            preds = agent.thinker.predict(state)
            preds_allowed_moves = [preds[idx] for idx in idxs]
            idstar = np.argmax(preds_allowed_moves)
            return possible_moves[idstar]
        else:
            rand_move = random.choice(possible_moves)
            return rand_move

def choose_boltzman(agent, state, possible_moves, current_episode = 0):
    """ Chooses the next move based on boltzman exploration """
    #print(f'Possible moves: {possible_moves}')
    tau = TAU_MAX * math.exp(-DECAY_RATE * current_episode)
    if possible_moves:
        idxs = [move[1] for move in possible_moves]
        
        preds = agent.policy_net.predict_value(state)
        probs = softmax([preds[idx] for idx in idxs], tau = tau)
        #print('\n'.join([f"{move[0]}; {prob}. Q: {preds[idx]}" for move, prob, idx in zip(possible_moves, probs, idxs)]))

        #print(f'Probs: {probs}')
        idx_star = np.random.choice(range(len(possible_moves)), p=probs)
        #print(f'Chosen move: {possible_moves[idx_star]}')
        return possible_moves[idx_star]
    
def softmax(q_values, tau=1.0):
    exp_values = np.exp(np.array(q_values) / tau)  # Scale by temperature
    probs = exp_values / np.sum(exp_values)
    return probs

def one_hot_encode(tile):
    one_hot = [0]*(len(ALL_NUMBERS)+1)
    idx = get_tile_idx(tile)
    one_hot[idx] = 1
    return one_hot

def get_tile_idx(tile):
    return TILE_MAPPING[number_to_str(tile.numbers)]


