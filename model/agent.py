from take_it_easy.board import Board
import random
from model.thinker import Thinker
from take_it_easy.value_functions import actual_score, score_line_smooth
from model.memory import  Memory, Experience
from model.exploration import EpsilonGreedyStrategy, ExplorationStrategy
import math
import numpy as np
import json
import torch
import itertools


def number_to_str(numbers: tuple[int]):
    return ''.join(str(n) for n in numbers)

ALL_NUMBERS = list(itertools.product([1,5,9], [3,4,8], [2,6,7]))
TILE_MAPPING = {number_to_str(n): i+1 for i, n in enumerate(ALL_NUMBERS)}
TILE_MAPPING[number_to_str((0,0,0))] = 0
print(TILE_MAPPING)

class AgentEasy:
    """ Agent that plays and learns the game.

        Attributes: 
            thinker (Thinker): the cognitive component of the agent, that can be trained and later on used to play.
            replay_memory (Memory): memory to remember transitions between states, used for learning.
            value_function: internal value function to rate the current boardstate. does not has to be the same as the overall game.
            n_actions: counter that keeps track of how many actions where performed.
    """

    def __init__(
            self,
            thinker: Thinker,
            memory: Memory,
            exploration_strategy: ExplorationStrategy,
            value_function = actual_score
            ):
        self.thinker = thinker
        self.replay_memory = memory
        self.exploration_strategy = exploration_strategy
        self.value_function = value_function
        self.n_actions = 0

    def translate_board(self, board: Board):
        """ Returns the state of the board.

        Creates the input array or features that serve as input to the neural net.
        Features are:
            * for each of the 19 tiles an array of length 28 with a 1 at the entry of tile that is present.
              last element is 0 if no tile is placed yet.
            * Dummy encoding for the tile currently at hand.
            * information about which pieces are left to be placed.
        
            Args:
                board (Board): boardstate
            Returns:
                list: one long list with bits containing the encoded boardstate.
        """
        x_input = []
        for row in board.tiles:
            for tile in row:
                x_input += self.one_hot_encode_tiles(tile)
        x_input += self.one_hot_encode_tiles(board.current_tile)
        x_input += self.binary_encode_pieces_left(board)
        return x_input

    def one_hot_encode_tiles(self, tile):
        ''' One hot encoding of a tile.'''
        one_hot = [0]*(len(ALL_NUMBERS)+1)
        idx = self.get_tile_idx(tile)
        one_hot[idx] = 1
        return one_hot
    
    @staticmethod
    def get_tile_idx(tile):
        ''' Lookup for position in the mapping dictionary. '''
        return TILE_MAPPING[number_to_str(tile.numbers)]
    
    def binary_encode_pieces_left(self, board: Board):
        """ Encodes the information of remaining tiles (excluding current piece).
        
        Args:
            board (Board): current boardstate
        
        Returns:
            list: list of 27 bits, with 1 if piece is still available and 0 otherwise.        
        """
        idxs_left = [self.get_tile_idx(tile) for tile in board.remaining_tiles]
        x_out = [1 if i in idxs_left else 0 for i in range(1, 28)]
        return x_out

    def act_and_observe_action(
            self,
            board: Board
            ):
        '''
            Does 3 things: observe boardstate before acting, act and observe boardstate after.
            The transition is saved as an Experience.
        '''
        state_t = self.translate_board(board)
        score_t = self.value_function(board)
        predictions = self.thinker.predict(state_t)
        loc, predicted_action = self.exploration_strategy.choose_next_move(
                                                          predictions,
                                                          possible_moves = board.get_open_moves()
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






