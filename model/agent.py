import itertools
from abc import ABC, abstractmethod

import torch

from model.board_representation import BoardEncoder
from model.exploration import IExplorationStrategy
from model.memory import Experience, IMemory
from model.thinker import DoubleQLearning
from take_it_easy.board import Board


def number_to_str(numbers: tuple[int]):
    return "".join(str(n) for n in numbers)


ALL_NUMBERS = list(itertools.product([1, 5, 9], [3, 4, 8], [2, 6, 7]))
TILE_MAPPING = {number_to_str(n): i + 1 for i, n in enumerate(ALL_NUMBERS)}
TILE_MAPPING[number_to_str((0, 0, 0))] = 0
print(TILE_MAPPING)


class ABCAgent(ABC):

    @abstractmethod
    def act(self, board: Board): ...

    @abstractmethod
    def learn(self): ...


class AgentEasy(ABCAgent):
    """Agent that plays and learns the game.

    Attributes:
        thinker (Thinker): the RL component of the agent..
        replay_memory (Memory): memory to remember transitions between states.
        n_actions: counter that keeps track of how many actions where performed.
    """

    def __init__(
        self,
        learner: DoubleQLearning,
        memory: IMemory,
        exploration_strategy: IExplorationStrategy,
        board_encoder: BoardEncoder,
    ):
        self.learner = learner
        self.replay_memory = memory
        self.exploration_strategy = exploration_strategy
        self.board_encoder = board_encoder
        self.n_actions = 0

    def act_and_observe_action(self, board: Board, greedy_action: bool = True):
        """
        Does 3 things: 
            1. observe boardstate before acting
            2. act
            3. observe boardstate after.
        The transition is saved as an Experience.
        """
        state_t, score_t, legal_moves_t, _ = self._observe_state(board)
        if greedy_action:
            action = self._act_greedily(board, state_t, legal_moves_t)
        else:
            action = self._act(board, state_t, legal_moves_t)
        state_t1, score_t1, legal_moves_t1, finished = self._observe_state(board)

        # ------------ Commit the Episode to Memory -------------- #
        reward = score_t1 - score_t
        legal_moves_mask = [1 if i in legal_moves_t1 else 0 for i in range(0, 19)]
        if not greedy_action:
            new_experience = Experience(
                state_t=torch.tensor(state_t, dtype=torch.float32),
                state_t1=torch.tensor(state_t1, dtype=torch.float32),
                action=torch.tensor(action, dtype=torch.int64),
                reward=torch.tensor(reward, dtype=torch.float32),
                finished=torch.tensor(finished, dtype=torch.bool),
                legal_moves=torch.tensor(legal_moves_mask, dtype=torch.long),
            )
            self.replay_memory.add(new_experience)
        self.n_actions += 1

    def learn(self):
        self.learner.learn_from_experience(self.replay_memory)

    def _act_greedily(self, board, state, legal_moves) -> int:
        predictions = self.learner.predict(state)
        action = max(legal_moves, key=lambda idx: predictions[idx])
        board.action_by_id(action)
        return action

    def _act(self, board, state, legal_moves) -> int:
        """makes a decision for the next move based on the current state."""

        predictions = self.learner.predict(state)
        action = self.exploration_strategy.choose_next_move(
            predictions, possible_moves=legal_moves
        )
        board.action_by_id(action)
        return action

    def _observe_state(self, board) -> tuple[list[float], int, list[int], bool]:
        """wrapper function that extracts all required data from current board state."""

        board_state = board.numeric_board_state()
        state = self.board_encoder.extract_features(board_state)
        score = board.current_score
        legal_moves = board.get_open_moves()
        finished = board.current_game_finished
        return state, score, legal_moves, finished

    def load_nnet(self):
        self.learner.policy_net = torch.load("model_checkpoint.pth")

    def save_nnet(self):
        self.learner.policy_net.save()
