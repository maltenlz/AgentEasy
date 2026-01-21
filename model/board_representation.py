import itertools
from abc import ABC, abstractmethod

import numpy as np


def number_to_str(numbers: tuple[int]):
    return "".join(str(n) for n in numbers)


def get_tile_by_idx(idx) -> list[int]:
    """returns list of numbers based on idx in the mapping"""
    tile_str = IDX_TO_TILE[number_to_str(idx)]
    return [int(s) for s in tile_str]


ALL_NUMBERS = list(itertools.product([1, 5, 9], [3, 4, 8], [2, 6, 7]))
TILE_TO_IDX = {number_to_str(n): i for i, n in enumerate(ALL_NUMBERS)}
IDX_TO_TILE = {i: number_to_str(n) for i, n in enumerate(ALL_NUMBERS)}


class BoardEncoder(ABC):
    """
    Given a fixed representation of the board, converts into features for learning.
    """

    @abstractmethod
    def extract_features(self, board_state: list[int]) -> np.array:
        """
        Expects an array of len 27 with the position of each of the tiles.
        -1: to be placed
        0: not yet placed.
        all others: position on the board, based on

        Returns:
            np.array: containing all the features that are fed into the neural network.
        """

    def get_input_shape(self):
        """returns the shape, used to initiate the nnet"""
        dummy_state = [0] * 27
        dummy_state[0] = -1
        features = self.extract_features(dummy_state)
        return len(features)


class SimpleOneHotEncoder(BoardEncoder):
    """
    Given a fixed representation of the board, converts into features for learning.
    """

    def extract_features(self, board_state: list[int]) -> np.array:
        x_out = []
        x_out += self._one_hot_encode_pieces(board_state)
        x_out += self._one_hot_encode_current_piece(board_state)
        x_out += self._one_hot_encode_pieces_left(board_state)
        return x_out

    def _one_hot_encode_pieces(self, board_state):
        """
        one hot encoding for each of the 20 spots on the board
        outdim: 19*27 = 513
        """
        x_out = []
        for board_spot in range(19):
            x_out += [1 if piece == board_spot else 0 for piece in board_state]
        return x_out

    def _one_hot_encode_current_piece(self, board_state):
        """
        one hot for piece currently in hand
        outdim: 27
        """
        matching_piece = [1 if piece == -2 else 0 for piece in board_state]
        return matching_piece

    def _one_hot_encode_pieces_left(self, board_state) -> list[int]:
        """
        one hot if piece is still out there
        outdim: 27
        """
        x_out = [1 if x == -1 else 0 for x in board_state]
        return x_out


class ColorPlusTileEncoder(SimpleOneHotEncoder):
    """
    Extension of the SimpleOneHotEncoder

    Also encodes the color of the tiles placed, 
    as well additional information about the tiles left.
    """

    def extract_features(self, board_state: list[int]) -> np.array:
        x_out = []
        x_out += self._one_hot_encode_pieces(board_state)
        x_out += self._one_hot_encode_current_piece(board_state)
        x_out += self._one_hot_encode_pieces_left(board_state)
        x_out += self._one_hot_encode_color(board_state)
        x_out += self._number_of_moves(board_state)
        return x_out

    def _one_hot_encode_color(self, board_state):
        """
        for each board position:
            create array of length 9 with 1 at the positions of the numbers.
        eg. 1,3,8 = [1,0,1,0,0,0,0,1,0]
        """
        x_out = []
        for board_spot in list(range(20)) + [-2]:
            x = [0] * 10
            try:
                tile_idx = board_state.index(board_spot)
                nbr_str = IDX_TO_TILE[tile_idx]
                for nbr in nbr_str:
                    x[int(nbr) - 1] = 1
                x_out += x
            except ValueError:
                x[-1] = 1
                x_out += x
        return x_out

    def _number_of_moves(self, board_state):
        """How many tiles are placed yet"""
        return [sum(1 if pos >= 0 else 0 for pos in board_state)]
