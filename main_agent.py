import pygame
import asyncio
import random 
from take_it_easy.constants import WIDTH, HEIGHT
from take_it_easy.board import Board
from take_it_easy.agent import AgentEasy
FPS = 60

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Take It Easy!")

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
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def transform_state(board):
    """ Transforms the board into a state that can be used by the agent """
    x_input = []
    for row in board.tiles:
        for tile in row:
            x_input += [1 if n in tile.numbers else 0 for n in range(0, 10)]
    x_input += [1 if n in board.current_tile.numbers else 0 for n in range(0, 10)]
    return x_input

def calculate_reward(board):
    """ Calculates the reward for the current state """
    return board.calculated_score

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
            if i > 50:
                agent.learn()
            # if i % 50 == 0:
            #     agent.synch_nets()
            i = i+1
        if board.tiles_placed == 19:
            plot_scores.append(board.calculated_score)
            mean_score = sum(plot_scores[-100:])/len(plot_scores[-100:])
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            board.refresh()
        board.draw_board()
        pygame.display.update()
    pygame.quit()

if __name__ == "__main__":
    main()

