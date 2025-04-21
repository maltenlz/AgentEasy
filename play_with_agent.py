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

4) Implement checkpoints to save the model every 5000 games or so ✘

5) Improve Convergence:
    5.1) Prioritized Experience Replay
    5.2) Learning Rate
        5.2.1) Higher Learning Rate
    5.3) In general tune hyperparameters
        5.3.1) Decrease Gamma
        5.3.2) Target network Tau
        5.3.3) Learning Rate
"""

import pygame
from take_it_easy.constants import WIDTH, HEIGHT
from take_it_easy.board import Board, ALL_NUMBERS
from take_it_easy.model.agent import AgentEasy
from take_it_easy.utils import transform_state
FPS = 60
import json

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Take It Easy!")



import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot_progress(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def main():
    run = True
    i = 30000
    clock = pygame.time.Clock()
    board = Board(WIN)
    agent = AgentEasy()
    agent.load_nnet()

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                board.action_by_mouse(event.pos)
                action, valid = board.action_by_mouse(event.pos)
                if valid == 'agent act':
                    agent.act(board, i)
            if event.type == pygame.MOUSEBUTTONUP:
                pass
            if event.type == pygame.QUIT:
                run = False

        board.draw_board()
        pygame.display.update()
    pygame.quit()

if __name__ == "__main__":
    main()
