import pygame
import math
from take_it_easy.constants import COLORS_NUMBERS, DIRECTION_NUMBERS, APOTHEM, GREY, BLACK, TILE_BACKGROUND


### utility functions to check if a point is inside a hexagon

def cross_product_2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def is_point_in_hexagon(point, hexagon):
    for i, edge in enumerate(hexagon):
        edge_start = edge
        edge_end = hexagon[(i + 1) % len(hexagon)]
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])
        point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])
        if cross_product_2d(edge_vector, point_vector) < 0:
            return False
    return True

### function that calculates the points of a hexagon

def calc_hexagon_points(center_x,center_y, apothem, scale):
    points = []
    radius = apothem * 2*3**0.5/3 * scale

    for _ in range(6):
        angle = _ * (2 * math.pi / 6)
        x = 100 + center_x + radius * math.cos(angle)
        y = 100 + center_y + radius * math.sin(angle)
        points.append((x, y))
    return points

### class that represents a tile, implements drawing and click handling

class Tile:
    def __init__(self, center_x, center_y, numbers: list = None) -> None:
        if numbers is None:
            numbers = [0,0,0]
        self.numbers = numbers
        self.points = calc_hexagon_points(center_x, center_y, APOTHEM, 0.97)
        self.points_boarder = calc_hexagon_points(center_x, center_y, APOTHEM, 1)
    
    def draw_tile(self, win: pygame.Surface) -> None:
        pygame.draw.polygon(win, BLACK, self.points_boarder)
        pygame.draw.polygon(win, TILE_BACKGROUND, self.points)
        for number in self.numbers:
            if number == 0:
                continue
            color = COLORS_NUMBERS[number]
            p1, p2 = DIRECTION_NUMBERS[number](self.points)
            pygame.draw.line(win, GREY, p1, p2, 16)
            pygame.draw.line(win, color, p1, p2, 14)

    def contains_point(self, point):
        return is_point_in_hexagon(point, self.points)

