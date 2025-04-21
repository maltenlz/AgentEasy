""" 
DDQN Agent that plays the game TakeItEasy.
Currently uses Double Q Learning with soft updates and gradient clipping.
Possible improvements:
1) try different representations of the tiles, currently no way of checking is certain tiles can still come
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

6) instead of predicting the action, construct all possible states after the action and predict the value of each state
    6.1) choose the action with the highest value
    6.2) probably much easier to learn, das model does not have to construct the board internally
    6.3) Function to generate the states given a tile
    6.4) adjust the nnet to single output state -> value
"""
import matplotlib.pyplot as plt
from IPython import display
import pygame
from take_it_easy.constants import WIDTH, HEIGHT
from take_it_easy.board import Board
from model.memory import SimpleReplayBuffer
from model.thinker import Thinker, EasyNet
from model.agent import AgentEasy
from model.exploration import EpsilonGreedyStrategy, ExplorationConfig

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 60

pygame.display.set_caption("Take It Easy!")




plt.ion()

def plot_progress(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Current')
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
    i = 0 
    clock = pygame.time.Clock()
    board = Board(WIN)
    
    thinker = Thinker(
            nnet_class = EasyNet
            )
    
    memory = SimpleReplayBuffer(capacity=200000)
    
    exploration_strategy = EpsilonGreedyStrategy(config = ExplorationConfig())

    agent = AgentEasy(
        thinker=thinker,
        memory = memory,
        exploration_strategy = exploration_strategy
        )

    # agent.study_games('molt_plays.json')
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
            agent.act_and_observe_action(board)
            if i > 21*19:
                agent.learn()
            i += 1
        if board.tiles_placed == 19:
            plot_scores.append(board.calculated_score)
            mean_score = sum(plot_scores[-100:])/len(plot_scores[-100:])
            plot_mean_scores.append(mean_score)
            if i % 100 == 0:
                plot_progress(plot_scores, plot_mean_scores)
            board.refresh()
        board.draw_board()
        if i % 20000 == 0:
            agent.save_nnet()            
        pygame.display.update()
    pygame.quit()

if __name__ == "__main__":
    main()
