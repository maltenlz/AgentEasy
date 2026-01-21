import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pygame

from model.agent import AgentEasy
from model.board_representation import ColorPlusTileEncoder
from model.exploration import (
    EpsilonGreedyStrategy,
    ExplorationConfig,
)
from model.memory import SimpleReplayBuffer
from model.thinker import DoubleQLearning, LearningConfig
from take_it_easy.board import Board
from take_it_easy.constants import HEIGHT, WIDTH

WIN = pygame.display.set_mode((WIDTH, HEIGHT))

SAVE_GREEDY_MOVES = False
pygame.display.set_caption("Take It Easy!")

plt.ion()


def check_if_converged(moving_avgs, window=10, warmup_periods=100, tolerance=5):
    """
    Heuristic to decide if performance converged and training can be safely stopped.

    Logic: if current avg is not better than the average over the last window periods 
    by at least tolerance -> converged.
    """
    if len(moving_avgs) < warmup_periods:
        return False
    longer_avg = sum(moving_avgs[-window - 1 : -1]) / window
    shorter_avg = sum(moving_avgs[-window // 10 :]) / (window // 10)
    return (shorter_avg - longer_avg) <= tolerance


if __name__ == "__main__":
    # Initiaize Board Encoder
    board_encoder = ColorPlusTileEncoder()

    # Initialize Double-Q-Learning
    double_q_learning = DoubleQLearning(
        learning_config=LearningConfig(
            lr=0.00001, nsteps_target_update=1, tau=0.01, batch_size=128, size_scaler=2
        )
    )
    double_q_learning.initialize_nnets(board_encoder.get_input_shape())
    double_q_learning.target_net.plot_nnet()

    memory = SimpleReplayBuffer(capacity=400_000)

    exploration_strategy = EpsilonGreedyStrategy(
        config=ExplorationConfig(
            epsilon0=0.7, epsilonmin=0.1, epsilon_decay_rate=200_000
        )
    )
    exploration_strategy.plot()

    agent = AgentEasy(
        learner=double_q_learning,
        memory=memory,
        exploration_strategy=exploration_strategy,
        board_encoder=board_encoder,
    )

    print(agent.board_encoder.get_input_shape())

    # Initialize the train loop
    game_scores = []
    mean_scores = []

    # Initialize mlflow
    mlflow.set_experiment("AgentEasy")
    start = time.time()

    with mlflow.start_run():
        n_games = 1
        mlflow.log_params(agent.learner.learning_config.__dict__)
        mlflow.log_params(agent.exploration_strategy.config.__dict__)
        mlflow.log_param(
            "Exploration Strategy", agent.exploration_strategy.__class__.__name__
        )
        mlflow.log_param("Save greey moves", SAVE_GREEDY_MOVES)
        mlflow.log_param("Memory Type", agent.replay_memory.__class__.__name__)
        mlflow.log_param("Board Representation", agent.board_encoder.__class__.__name__)
        num_params = sum(p.numel() for p in agent.learner.policy_net.parameters())
        mlflow.log_param("Nnet Parameters", num_params)
        mlflow.log_artifact("nnet_architecture.png")
        mlflow.log_artifact("visualisations/exploration_schedule.png")
        # Initialize the game
        clock = pygame.time.Clock()
        board = Board(WIN)

        # Start the trainloop
        while True:
            if not board.current_game_finished:
                make_greedy_move = (n_games % 100) >= 90
                agent.act_and_observe_action(board, make_greedy_move)
                if n_games > 10:
                    agent.learn()
            else:
                game_scores.append(board.current_score)
                if n_games % 100 == 0:
                    last_10_games = np.array(game_scores[-10:])
                    mean_score = np.mean(last_10_games)
                    mlflow.log_metric("Mean Score", mean_score, step=n_games)
                board.reset_game()
                n_games += 1
                if n_games % 1000 == 0:
                    agent.save_nnet()
                    end = time.time()
                    print(f"Time for {n_games} games: {end - start:.2f} seconds")

            if n_games == 300_000:
                break
            # board.render_board()
            # pygame.display.update()
        pygame.quit()
