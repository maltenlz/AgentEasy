import matplotlib.pyplot as plt
from IPython import display
import pygame
from take_it_easy.constants import WIDTH, HEIGHT
from take_it_easy.board import Board
from model.memory import SimpleReplayBuffer, PERBuffer
from model.thinker import Thinker, EasyNet, LearningConfig
from model.agent import AgentEasy
from model.exploration import EpsilonGreedyStrategy, BoltzmannExploration, ExplorationConfig
from model.value_functions import smooth_score, actual_score
import mlflow
import subprocess
import numpy as np
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 60

pygame.display.set_caption("Take It Easy!")


def get_git_commit():
    ''' gets the current git commit '''
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

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

def check_if_converged(moving_avgs, window=10, warmup_periods=100, tolerance=5):
    ''' 
    Heuristic to decide if performance converged and training can be safely stopped. 
    
    Logic: if current avg is not better than the average over the last window periods by at least tolerance -> converged.
    '''
    if len(moving_avgs) < warmup_periods:
        return False
    longer_avg = sum(moving_avgs[-window-1:-1]) / window
    shorter_avg = sum(moving_avgs[-window//10:]) / (window//10)
    return (shorter_avg - longer_avg) <= tolerance



if __name__ == "__main__":


    # Also intialize the Agent
    thinker = Thinker(
        nnet_class = EasyNet,
        learning_config=LearningConfig(lr = 0.00001, nsteps_target_update=1, tau=0.01, batch_size=128)
        )
    thinker.target_net.plot_nnet()
    memory = PERBuffer(capacity=2000000)
    exploration_strategy = EpsilonGreedyStrategy(config = ExplorationConfig(eps_start=1, eps_end=1, eps_decay=400000))
    exploration_strategy.plot()
    agent = AgentEasy(
    thinker=thinker,
    memory=memory,
    exploration_strategy=exploration_strategy,
    value_function=actual_score
    )

    # Initialize the train loop
    game_scores = []
    mean_scores = []

    # Initialize mlflow
    mlflow.set_experiment("AgentEasy")
    with mlflow.start_run():

        i = 0
        mlflow.log_params(agent.thinker.learning_config.__dict__)
        mlflow.log_params(agent.exploration_strategy.config.__dict__)
        mlflow.log_param('Exploration Strategy', agent.exploration_strategy.strategy_name)
        mlflow.log_param('Memory Type', agent.replay_memory.memory_type)

        mlflow.log_param('Nnet Parameters', len([p for p in agent.thinker.policy_net.parameters()]))
        mlflow.log_artifact("nnet_architecture.png")

        # Initialize the game
        clock = pygame.time.Clock()
        board = Board(WIN)
        converged = False

        # Start the trainloop
        while not converged:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    board.action_by_mouse(event.pos)
                if event.type == pygame.MOUSEBUTTONUP:
                    pass
                if event.type == pygame.QUIT:
                    break
            game_not_finished = board.tiles_placed < 19
            if game_not_finished:
                agent.act_and_observe_action(board)
                if i > 21*19:
                    agent.learn()
                i += 1
            else:
                game_scores.append(board.calculated_score)
                last_100_games = np.array(game_scores[-100:])

                mean_score = np.mean(last_100_games)
                median_score = np.median(last_100_games)
                p_10_score = np.quantile(last_100_games, q=0.1)
                p_90_score = np.quantile(last_100_games, q=0.9)

                mean_scores.append(mean_score)
                if i % 100 == 0:
                    plot_progress(game_scores, mean_scores)
                    mlflow.log_metric("Mean Score", mean_score, step=i)
                    mlflow.log_metric("Median Score", median_score, step=i)
                    mlflow.log_metric("P10 Score", p_10_score, step=i)
                    mlflow.log_metric("P90 Score", p_90_score, step=i)
                board.refresh()
            if i == 4_000_000:
                break
            board.draw_board()
            if i % 20000 == 0:
                agent.save_nnet()
            pygame.display.update()
        pygame.quit()