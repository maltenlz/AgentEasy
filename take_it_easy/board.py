import pygame
import itertools
import random

from take_it_easy.constants import WHITE, BLACK, WIDTH, HEIGHT, BOARD_HEIGHT, TILE_BACKGROUND
from take_it_easy.tile import Tile
from take_it_easy.value_functions import board_value, score_line, score_line_smooth
pygame.font.init()

### Lookup tables to get the ordering correct
TILE_POSITIONS = [[(0,1), (0,2), (0,3)],
                  [(1,1), (1,2), (1,3), (1,4)],
                  [(2,0), (2,1), (2,2), (2,3), (2,4)],
                  [(3,1), (3,2), (3,3), (3,4)],
                  [(4,1), (4,2), (4,3)]
                 ]

### Lookup tables to calculate the score

ALL_NUMBERS = list(itertools.product([1,5,9], [3,4,8], [2,6,7]))

### utility functions to calculate the center points of the tiles, to know where to draw them -> to tile class again?
def get_center_points(i, j, height):
    center_x = i * (height/5) /( 2*3**0.5/3) + (height/5) / 2
    center_y = j * (height/5)
    if not i % 2:
        center_y += (height//5) / 2
    return center_x, center_y

#############################################
#### 1. Refactor more into tiles
#### 2. Show remaining tiles
#### 3. save scores
#############################################

class Board:
    """ Implements the game logic and drawing of the board """
    def __init__(self, win) -> None:
        self.win = win
        self.refresh()

    def _create_pool(self) -> None:
        ### POSITIONS of the remaining tiles
        REMAINING_POSIONS = ([(6,i) for i in range(6)] +
                            [(7,i) for i in range(6)] +
                            [(8,i) for i in range(6)] + 
                            [(9,i) for i in range(6)] + 
                            [(10,i) for i in range(6)])
        self.remaining_tiles = []
        self.open_numbers = ALL_NUMBERS
        for number in self.open_numbers:
            tile = Tile(*get_center_points(*REMAINING_POSIONS.pop(), BOARD_HEIGHT))
            tile.numbers = number
            self.remaining_tiles.append(tile)

    def _next_tile(self) -> None:
        self.current_tile = random.choice(self.remaining_tiles)
        self.remaining_tiles.remove(self.current_tile)
        self.tiles_placed += 1
        next_tile_layout = Tile(WIDTH * 0.42, HEIGHT * 0.01)
        self.current_tile.points = next_tile_layout.points
        self.current_tile.points_boarder = next_tile_layout.points_boarder
        self._calculate_score()
        if self.tiles_placed == 19:
            self.finished = True

    def refresh(self) -> None:
        self.tiles = [[Tile(*get_center_points(tile[0], tile[1], BOARD_HEIGHT)) for tile in row] for row in TILE_POSITIONS]
        self.pieces_left = []
        self.calculated_score = 0
        self.selected_tile = None
        self.tiles_placed = -1
        self.finished = False
        self._create_pool()
        self._next_tile()
        self.refresh_button = pygame.Rect(WIDTH * 0.42, HEIGHT * 0.9, 130, 80)

    def action_by_mouse(self, pos: tuple) -> None:
        idx = 0
        for col in self.tiles:
            for tile in col:
                if tile.contains_point(pos) and tile.numbers == [0,0,0]:
                    tile.numbers = self.current_tile.numbers
                    self._next_tile()
                    return idx, True
                idx += 1
        if self.refresh_button.collidepoint(*pos):
            self.refresh()
        return None, False
                
    def action_by_id(self, pos: tuple) -> bool:
        """ for interaction with ai """
        if self.tiles[pos[0]][pos[1]].numbers == [0,0,0]:
            self.tiles[pos[0]][pos[1]].numbers = self.current_tile.numbers
            self._next_tile()
        else:
            raise ValueError("Tile already placed")

    def draw_board(self) -> None:
        self.win.fill(WHITE)
        for col in self.tiles:
            for tile in col:
                tile.draw_tile(self.win)
        for tile in self.remaining_tiles:
            tile.draw_tile(self.win)
        if self.tiles_placed < 20:
            self.current_tile.draw_tile(self.win)
        pygame.draw.line(self.win, BLACK, (WIDTH * 0.55, 0), (WIDTH * 0.55, HEIGHT), 4)
        self._draw_refresh()
        if self.tiles_placed == 19:
            self._draw_score()

    def _draw_refresh(self):
        font_size = 24
        text = "Refresh"
        pygame.draw.rect(self.win, TILE_BACKGROUND, self.refresh_button)
        font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.refresh_button.center)
        self.win.blit(text_surface, text_rect)

    def _calculate_score(self) -> None:
        self.calculated_score = board_value(self, score_line)
        self.smooth_value = board_value(self, score_line_smooth)

    def _draw_score(self):
        font = pygame.font.SysFont("helvetica", 100, bold = True)
        text_render = font.render(f"SCORE: {self.calculated_score}", True, BLACK)
        text_rect = text_render.get_rect(center=(BOARD_HEIGHT // 2 + 80, BOARD_HEIGHT // 2 + 80))
        self.win.blit(text_render, text_rect)