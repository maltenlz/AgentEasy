import itertools

ALL_NUMBERS = list(itertools.product([1,5,9], [3,4,8], [2,6,7]))

def number_to_str(numbers: tuple[int]):
    return ''.join(str(n) for n in numbers)

TILE_MAPPING = {number_to_str(n): i+1 for i, n in enumerate(ALL_NUMBERS)}
TILE_MAPPING[number_to_str((0,0,0))] = 0

def one_hot_encode(tile):
    one_hot = [0]*(len(ALL_NUMBERS)+1)
    idx = get_tile_idx(tile)
    one_hot[idx] = 1
    return one_hot

def get_tile_idx(tile):
    return TILE_MAPPING[number_to_str(tile.numbers)]

def transform_state(board):
    """ Transforms the board into a state that can be used by the agent """
    x_input = []
    for row in board.tiles:
        for tile in row:
            x_input += one_hot_encode(tile)
    x_input += one_hot_encode(board.current_tile)

    # Also encode colors to make it easier
    for row in board.tiles:
        for tile in row:
            x_input += [1 if number in tile.numbers else 0 for number in range(10)]
    x_input += [1 if number in board.current_tile.numbers else 0 for number in range(10)]

    # Also encode the remaining tiles
    remaining_numbers = [number_to_str(tile.numbers) for tile in board.remaining_tiles]
    x_input += [1 if key in remaining_numbers else 0 for key, value in TILE_MAPPING.items()]
    x_input += [len(remaining_numbers)]
    return x_input

ALL_NUMBERS = list(itertools.product([1,5,9], [3,4,8], [2,6,7]))
