from typing import Any
from take_it_easy.board import Board
import time
import random
from collections import deque
from take_it_easy.model import EasyNet, QTrainer
import math
import numpy as np

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 40000
MAX_MEMORY = 10000000

class AgentEasy:
    """ Agent that plays and learns the game."""

    def __init__(self):
        self.state_dim = 19
        self.memory = deque(maxlen=MAX_MEMORY)
        self.policy_net = EasyNet()
        self.target_net = EasyNet()
        self.trainer = QTrainer(self.policy_net, self.target_net)

    def act(self, state, board: Board, steps_done = 0):
        """ Dummy agent that just places a random tile on a random position"""
        possible_moves = self._get_open_moves(board)
        if possible_moves:
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            if sample > eps_threshold and steps_done > 50:
                idxs = [move[1] for move in possible_moves]
                preds = self.policy_net.predict_value(state)
                idstar = np.argmax([preds[idx] for idx in idxs])
                return possible_moves[idstar]
            else:
                rand_move = random.choice(possible_moves)
                return rand_move

    def chache(self, state_t, action, state_t1, reward, finished):
        """ Dummy agent does not learn """
        self.memory.append((state_t, action, state_t1, reward, finished))
    
    def learn(self):
        batch = random.choices(self.memory, k = 1024)
        self.trainer.train_step(*zip(*batch))

    def save_nnet(self):
        self.policy_net.save()
        
    def synch_nets(self):
        self.trainer.update_target()

    def _get_open_moves(self, board: Board):
        """ Returns all possible moves on the board """
        moves = []
        idx = 0
        for i, row in enumerate(board.tiles):
            for j, tile in enumerate(row):
                if tile.numbers == [0, 0, 0]:
                    moves.append([(i,j), idx])
                idx += 1
        return moves