import pygame
import itertools
import random

from typing import List

from take_it_easy.constants import WHITE, BLACK, WIDTH, HEIGHT, BOARD_HEIGHT, TILE_BACKGROUND, STRAIGHT_LINES, DIAGS_LEFT, DIAGS_RIGHT
from take_it_easy.tile import Tile

# ----------------------------- Utility functions for scoring ------------------------------------- #
def score_straight(board, scoring_func):
    return sum(scoring_func([board.board_tiles[pos[0]][pos[1]].numbers[0] for pos in line]) for line in STRAIGHT_LINES)

def score_left_diags(board, scoring_func):
    return sum(scoring_func([board.board_tiles[pos[0]][pos[1]].numbers[1] for pos in line]) for line in DIAGS_LEFT)

def score_right_diags(board, scoring_func):
    return sum(scoring_func([board.board_tiles[pos[0]][pos[1]].numbers[2] for pos in line]) for line in DIAGS_RIGHT)

def score_line(numbers:list):
    """ only awards points if only a single color is present """
    if len(set(numbers)) == 1:
        return sum(numbers)
    return 0

def actual_score(board, **kwargs):
    return score_straight(board, score_line) + score_left_diags(board, score_line) + score_right_diags(board, score_line)


# ----------------------------- Utility functions for scoring ------------------------------------- #
def number_to_str(numbers: tuple[int]):
    return ''.join(str(n) for n in numbers)

def get_tile_idx(tile):
        ''' Lookup for position in the mapping dictionary. '''
        return TILE_TO_IDX[number_to_str(tile.numbers)]

ALL_NUMBERS = list(itertools.product([1,5,9], [3,4,8], [2,6,7]))
TILE_TO_IDX = {number_to_str(n): i for i, n in enumerate(ALL_NUMBERS)}
IDX_TO_TILE = {i: number_to_str(n) for i, n in enumerate(ALL_NUMBERS)}

pygame.font.init()

### Lookup tables to get the ordering correct
TILE_POSITIONS = [
                  [(0,1), (0,2), (0,3)],
                  [(1,1), (1,2), (1,3), (1,4)],
                  [(2,0), (2,1), (2,2), (2,3), (2,4)],
                  [(3,1), (3,2), (3,3), (3,4)],
                  [(4,1), (4,2), (4,3)]
                 ]

def create_idx_pos_mapping():
    mapping = {}
    idx = 0
    for i, row in enumerate(TILE_POSITIONS):
        for j, _ in enumerate(row):
            mapping[idx] = (i, j)
            idx += 1
    return mapping

IDX_TO_POS = create_idx_pos_mapping()

def get_center_points(i, j, height):
    center_x = i * (height/5) /( 2*3**0.5/3) + (height/5) / 2
    center_y = j * (height/5)
    if not i % 2:
        center_y += (height//5) / 2
    return center_x, center_y

# -------------------- Precomputing draw pile and empty board for efficiency -------------------------------- #
def generate_draw_pile() -> List[Tile]:
    """ resets the draw pile """
    all_positions = (
                        [(6,i) for i in range(6)] +
                        [(7,i) for i in range(6)] +
                        [(8,i) for i in range(6)] +
                        [(9,i) for i in range(6)] +
                        [(10,i) for i in range(6)]
                        )
    remaining_tiles = []
    for pos, number in zip(all_positions, ALL_NUMBERS):
        tile = Tile(*get_center_points(*pos, BOARD_HEIGHT))
        tile.numbers = number
        remaining_tiles.append(tile)
    return remaining_tiles

DRAW_PILE = generate_draw_pile()
EMPTY_BOARD = [
                [Tile(*get_center_points(tile[0], tile[1], BOARD_HEIGHT)) for tile in row]
                for row in TILE_POSITIONS
              ]

class Board:
    """ Implements the game logic and drawing of the board """

    def __init__(self, win: pygame.display) -> None:
        self.win = win
        self.remaining_tiles = []
        self.board_tiles = []
        self.current_tile = []

        # static buttons
        self.refresh_button = pygame.Rect(WIDTH * 0.42, HEIGHT * 0.9, 130, 80)
        self.act_button = pygame.Rect(WIDTH * 0.01, HEIGHT * 0.9, 130, 80)

        # reset game to initalize
        self.reset_game()

    def numeric_board_state(self) -> List[int]:
        ''' 
        returns an array that fully describes the board state
        for each of the 27 tiles,
        -1 if not yet placed
        -2 if its the tile to be placed next
        and otherwise the index of the spot on the board.
        will be used by the BoardRepresentation.
        '''
        out = [-1]*27
        flattened_tiles = [item for sublist in self.board_tiles for item in sublist]
        for j, tile in enumerate(flattened_tiles):
            if tile.numbers == [0,0,0]:
                continue
            out[get_tile_idx(tile)] = j
        out[get_tile_idx(self.current_tile)] = -2
        return out

    def reset_game(self):
        """ Reset the game state """
        self._reset_board()
        self._reset_draw_pile()
        self._draw_next_tile()

    def _reset_draw_pile(self):
        """ resets the draw pile """
        self.remaining_tiles =[tile.copy() for tile in DRAW_PILE]

    def _reset_board(self):
        """ resets the board """
        self.board_tiles = [[tile.copy() for tile in row] for row in EMPTY_BOARD]

    def _draw_next_tile(self):
        self.current_tile = random.choice(self.remaining_tiles)
        self.remaining_tiles.remove(self.current_tile)

    def action_by_mouse(self, pos: tuple):
        idx = 0
        for col in self.board_tiles:
            for tile in col:
                if tile.contains_point(pos) and tile.numbers == [0,0,0]:
                    tile.numbers = self.current_tile.numbers
                    self._draw_next_tile()
                    return idx, 'manual place'
                idx += 1
        if self.refresh_button.collidepoint(*pos):
            self.reset_game()
        if self.act_button.collidepoint(*pos):
            return None, 'agent act'
        return None, 'invalid click'
                
    def action_by_id(self, action_id: int):
        """ places tile based on ID """
        i, j = IDX_TO_POS[action_id]
        tile = self.board_tiles[i][j]
        if tile.numbers == [0,0,0]:
            tile.numbers = self.current_tile.numbers
            self._draw_next_tile()
        else:
            raise ValueError("Tile already placed")
    
    def get_open_moves(self) -> List[int]:
        """ Returns all possible moves on the board """
        moves = []
        idx = 0
        for i, row in enumerate(self.board_tiles):
            for j, tile in enumerate(row):
                if tile.numbers == [0, 0, 0]:
                    moves.append(idx)
                idx += 1
        return moves
    
    def render_board(self) -> None:

        self.win.fill(WHITE)
        for col in self.board_tiles:
            for tile in col:
                tile.render_tile(self.win)

        for tile in self.remaining_tiles:
            tile.render_tile(self.win)

        if self.tiles_placed < 20:
            self.current_tile.change_location_to_next_tile()
            self.current_tile.render_tile(self.win)

        pygame.draw.line(self.win, BLACK, (WIDTH * 0.55, 0), (WIDTH * 0.55, HEIGHT), 4)

        self._render_refresh_button()
        self._render_agent_act_button()

        if self.tiles_placed == 19:
            self._render_score()
    
    def _render_refresh_button(self):
        """ refresh button to reset the game """
        font_size = 24
        text = "Refresh"
        pygame.draw.rect(self.win, TILE_BACKGROUND, self.refresh_button)
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.refresh_button.center)
        self.win.blit(text_surface, text_rect)

    def _render_agent_act_button(self):
        """ clicking will prompt agent to take an action. """
        font_size = 24
        text = "Agent Act"
        pygame.draw.rect(self.win, TILE_BACKGROUND, self.act_button)
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.act_button.center)
        self.win.blit(text_surface, text_rect)

    def _render_score(self):
        font = pygame.font.SysFont("helvetica", 100, bold = True)
        text_render = font.render(f"SCORE: {self.current_score}", True, BLACK)
        text_rect = text_render.get_rect(center=(BOARD_HEIGHT // 2 + 80, BOARD_HEIGHT // 2 + 80))
        self.win.blit(text_render, text_rect)
    
    @property
    def current_game_finished(self) -> bool:
        """ current game is finished? """
        return self.tiles_placed == 19
    
    @property
    def current_score(self) -> None:
        return actual_score(self)

    @property
    def tiles_placed(self):
        """ returns the number of tiles still in the pool """
        return 27 - len(self.remaining_tiles) - 1
