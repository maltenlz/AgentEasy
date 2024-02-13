import pygame
from take_it_easy.constants import WIDTH, HEIGHT
from take_it_easy.board import Board, ALL_NUMBERS
from take_it_easy.agent import AgentEasy
FPS = 60

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Take It Easy!")
""" 
DDQN Agent that plays the game TakeItEasy.
Currently uses Double Q Learning with soft updates and gradient clipping.
Possible improvements:
1) try different representations of the tiles, currently no way of checking is certain tiles can still come ✘
27 tiles. therefore 28 one-hoty-encoding for each field. -> harder to learn but much more potential ✘

2) kickstart learning by providing a set of decent moves ~ 100 games or so
    2.1) seperapte script that captures games
    2.2) save games to disk in appropriate format

3) prepare PC for training -> GPU -> increase size!  ✘
    3.1) Increase network size!  ✘
    3.2) Increase batch size!  ✘

4) Implement checkpoints to save the model every 5000 games or so
"""

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    #plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    #plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


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
    # Also encode the remaining tiles
    remaining_numbers = [number_to_str(tile.numbers) for tile in board.remaining_tiles]
    x_input += [1 if key in remaining_numbers else 0 for key, value in TILE_MAPPING.items()]
    x_input += [len(remaining_numbers)]
    return x_input

def main():
    run = True
    i = 0 
    clock = pygame.time.Clock()
    board = Board(WIN)
    agent = AgentEasy()
    plot_scores = []
    plot_mean_scores = []
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                board.action_by_mouse(event.pos)
            if event.type == pygame.MOUSEBUTTONUP:
                pass
            if event.type == pygame.QUIT:
                run = False
        if board.tiles_placed <= 19:
            state_t = transform_state(board)
            choice = agent.act(state_t, board, i)
            score_t = board.smooth_value
            board.action_by_id(choice[0])
            state_t1 = transform_state(board)
            score_t1 = board.smooth_value
            reward = score_t1 - score_t
            agent.chache(state_t, choice[1], state_t1, reward, board.finished)
            agent.synch_nets()
            if i > 128*19:
                agent.learn()
            # if i % 250 == 0:
                
            i = i+1
        if board.tiles_placed == 19:
            plot_scores.append(board.calculated_score)
            mean_score = sum(plot_scores[-100:])/len(plot_scores[-100:])
            plot_mean_scores.append(mean_score)
            if i % 10 == 0:
                plot(plot_scores, plot_mean_scores)
            board.refresh()
        #board.draw_board()
        if i % 200 == 0:
            agent.save_nnet()
        #pygame.display.update()
    pygame.quit()

if __name__ == "__main__":
    main()

