from take_it_easy.constants import STRAIGHT_LINES, DIAGS_LEFT, DIAGS_RIGHT
DISOUNT_FACTOR = 0.5

### utility functions to calcualte points at the end of the game
def score_straight(board, scoring_func):
    return sum(scoring_func([board.tiles[pos[0]][pos[1]].numbers[0] for pos in line]) for line in STRAIGHT_LINES)

def score_left_diags(board, scoring_func):
    return sum(scoring_func([board.tiles[pos[0]][pos[1]].numbers[1] for pos in line]) for line in DIAGS_LEFT)

def score_right_diags(board, scoring_func):
    return sum(scoring_func([board.tiles[pos[0]][pos[1]].numbers[2] for pos in line]) for line in DIAGS_RIGHT)

def score_line(numbers:list):
    if len(set(numbers)) == 1:
        return sum(numbers)
    return 0

def score_line_smooth(numbers:list):
    non_zeroes = [n for n in numbers if n != 0]
    if len(set(non_zeroes)) == 1:
        if len(non_zeroes) == len(numbers):
            return sum(non_zeroes)
        else:
            return sum(non_zeroes) * DISOUNT_FACTOR
    return 0

def board_value(board, scoring_func):
    return score_straight(board, scoring_func) + score_left_diags(board, scoring_func) + score_right_diags(board, scoring_func)
