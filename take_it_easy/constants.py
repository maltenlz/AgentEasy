HEIGHT = 900
WIDTH = HEIGHT / (2 / (3**0.5)) + 600

BOARD_HEIGHT = HEIGHT - 250
BOARD_WIDTH = HEIGHT - 1900

COLS = ROWS = 5
APOTHEM = (BOARD_HEIGHT) / (ROWS * 2)
RADIUS = APOTHEM * 2 * 3**0.5 / 3

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TILE_BACKGROUND = (112, 128, 144)
GREY = (128, 128, 128)

COLORS_NUMBERS = {
    1: (192, 192, 192),  # grey
    2: (255, 182, 193),  # pink light
    3: (255, 105, 180),  # pink
    4: (0, 255, 255),  # light blue
    5: (135, 206, 235),  # turquoise
    6: (220, 20, 60),  # red
    7: (50, 205, 50),  # green
    8: (255, 127, 80),  # orange
    9: (255, 215, 0),  # yellow
}

STRAIGHT_LINES = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
    [(3, 0), (3, 1), (3, 2), (3, 3)],
    [(4, 0), (4, 1), (4, 2)],
]

DIAGS_RIGHT = [
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 0)],
    [(0, 2), (1, 2), (2, 2), (3, 1), (4, 0)],
    [(1, 3), (2, 3), (3, 2), (4, 1)],
    [(2, 4), (3, 3), (4, 2)],
]

DIAGS_LEFT = [
    [(2, 4), (1, 3), (0, 2)],
    [(3, 3), (2, 3), (1, 2), (0, 1)],
    [(4, 2), (3, 2), (2, 2), (1, 1), (0, 0)],
    [(4, 1), (3, 1), (2, 1), (1, 0)],
    [(4, 0), (3, 0), (2, 0)],
]


def straight(points):
    x1 = (points[4][0] + points[5][0]) / 2
    y1 = (points[4][1] + points[5][1]) / 2
    x2 = (points[1][0] + points[2][0]) / 2
    y2 = (points[1][1] + points[2][1]) / 2
    return (x1, y1), (x2, y2)


def left_diag(points):
    x1 = (points[0][0] + points[1][0]) / 2
    y1 = (points[0][1] + points[1][1]) / 2
    x2 = (points[3][0] + points[4][0]) / 2
    y2 = (points[3][1] + points[4][1]) / 2
    return (x1, y1), (x2, y2)


def right_diag(points):
    x1 = (points[0][0] + points[5][0]) / 2
    y1 = (points[0][1] + points[5][1]) / 2
    x2 = (points[2][0] + points[3][0]) / 2
    y2 = (points[2][1] + points[3][1]) / 2
    return (x1, y1), (x2, y2)


DIRECTION_NUMBERS = {
    1: straight,
    2: right_diag,
    3: left_diag,
    4: left_diag,
    5: straight,
    6: right_diag,
    7: right_diag,
    8: left_diag,
    9: straight,
}
