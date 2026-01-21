from take_it_easy.constants import DIAGS_LEFT, DIAGS_RIGHT, STRAIGHT_LINES

DISOUNT_FACTOR = 0.2


### utility functions to calcualte points at the end of the game
def score_straight(board, scoring_func):
    return sum(
        scoring_func([board.tiles[pos[0]][pos[1]].numbers[0] for pos in line])
        for line in STRAIGHT_LINES
    )


def score_left_diags(board, scoring_func):
    return sum(
        scoring_func([board.tiles[pos[0]][pos[1]].numbers[1] for pos in line])
        for line in DIAGS_LEFT
    )


def score_right_diags(board, scoring_func):
    return sum(
        scoring_func([board.tiles[pos[0]][pos[1]].numbers[2] for pos in line])
        for line in DIAGS_RIGHT
    )


def score_line(numbers: list):
    if len(set(numbers)) == 1:
        return sum(numbers)
    return 0


def score_line_smooth(numbers: list):
    non_zeroes = [n for n in numbers if n != 0]
    # only one number
    numbers_placed = len(non_zeroes)
    line_length = len(numbers)
    if len(set(non_zeroes)) == 1:
        # completly filled
        if numbers_placed == line_length:
            return sum(non_zeroes)
        else:
            # also reward continuing lines
            return sum(non_zeroes) * numbers_placed / line_length * DISOUNT_FACTOR
    return 0


def actual_score(board, scoring_func=score_line):
    return (
        score_straight(board, scoring_func)
        + score_left_diags(board, scoring_func)
        + score_right_diags(board, scoring_func)
    )
