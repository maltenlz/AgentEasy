import itertools
import json


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

def save_transition_to_json(filename, data_dict):
    """ saves the transition to a json file """
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
            existing_data.update(data_dict)  # Update if the file exists
        with open(filename, 'w') as f:
            json.dump(existing_data, f)
    except FileNotFoundError:
        with open(filename, 'w') as f:  # Write if the file doesn't exist
            json.dump(data_dict, f)

ALL_NUMBERS = list(itertools.product([1,5,9], [3,4,8], [2,6,7]))
