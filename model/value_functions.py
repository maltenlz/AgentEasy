from take_it_easy.constants import STRAIGHT_LINES, DIAGS_LEFT, DIAGS_RIGHT
DISOUNT_FACTOR = 0.4

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
    ''' 
        If line is full return the full value.
        If instead it is a partial line, use the sum of current numbers times a discount factors.
        Idea is to rewards putting same numbers in lines
    '''
    non_zeroes = [n for n in numbers if n != 0]
    numbers_placed = len(non_zeroes)
    line_length = len(numbers)
    if len(set(non_zeroes)) == 1:
        if numbers_placed == line_length:
            # completed line
            return sum(non_zeroes)
        return sum(non_zeroes) * DISOUNT_FACTOR
    return 0

def actual_score(board, **kwargs):
    return score_straight(board, score_line) + score_left_diags(board, score_line) + score_right_diags(board, score_line)

def smooth_score(board, **kwargs):
    return score_straight(board, score_line_smooth) + score_left_diags(board, score_line_smooth) + score_right_diags(board, score_line_smooth)

def mixture_score(board, **kwargs):
    mixing_factor = 0.5
    return mixing_factor * actual_score(board) + (1 - mixing_factor) * smooth_score(board)
