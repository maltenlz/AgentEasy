from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import numpy as np
import random
@dataclass
class ExplorationConfig:
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay: float = 150000

class ExplorationStrategy(ABC):
    def __init__(
            self, 
            config: ExplorationConfig
            ):
        self.n_steps = 0
        self.config = config

    @abstractmethod
    def choose_next_move(
        self,
        predictions,
        possible_moves
        ):
        ''' Stratgegy to decide on the next move '''

def increment_steps(method):
    ''' utility function to increase step counter for the ExplorationStrategy Class'''
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.n_steps += 1
        return result
    return wrapper

class EpsilonGreedyStrategy(ExplorationStrategy):
    ''' Implementation of Epsilon Greedy Move '''
    @property
    def eps_threshold(self):
        ''' Calculates the chance of taking a random move at a given time '''
        return self.config.eps_end + (self.config.eps_start - self.config.eps_end) * math.exp(-1. * self.n_steps / self.config.eps_decay)
    
    @increment_steps
    def choose_next_move(
                        self,
                        predictions,
                        possible_moves
                        ):
        if possible_moves:
            random_draw = random.random()
            if random_draw > self.eps_threshold:
                idxs = [move[1] for move in possible_moves]
                preds_allowed_moves = [predictions[idx] for idx in idxs]
                idstar = np.argmax(preds_allowed_moves)
                return possible_moves[idstar]
            else:
                rand_move = random.choice(possible_moves)
                return rand_move

# def choose_boltzman(agent, state, possible_moves, current_episode = 0):
#     """ Chooses the next move based on boltzman exploration """
#     #print(f'Possible moves: {possible_moves}')
#     tau = TAU_MAX * math.exp(-DECAY_RATE * current_episode)
#     if possible_moves:
#         idxs = [move[1] for move in possible_moves]
        
#         preds = agent.policy_net.predict_value(state)
#         probs = softmax([preds[idx] for idx in idxs], tau = tau)
#         #print('\n'.join([f"{move[0]}; {prob}. Q: {preds[idx]}" for move, prob, idx in zip(possible_moves, probs, idxs)]))

#         #print(f'Probs: {probs}')
#         idx_star = np.random.choice(range(len(possible_moves)), p=probs)
#         #print(f'Chosen move: {possible_moves[idx_star]}')
#         return possible_moves[idx_star]
    
