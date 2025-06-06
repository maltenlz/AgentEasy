from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
import random
import matplotlib.pyplot as plt


def increment_steps(method):
    ''' Utility function to increase step counter for the ExplorationStrategy Class'''
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.n_steps += 1
        return result
    return wrapper
@dataclass
class ExplorationConfig:
    ''' Container class to store the hyperparameters '''
    eps_start: Optional[float] = 0.5
    eps_end: Optional[float] = 0.05
    eps_decay: Optional[float] = 150000
    tau0: Optional[float] = 0.5
    taumin: Optional[float] = 0.01
    tau_decay_rate: Optional[float] = 0.0001
class ExplorationStrategy(ABC):
    ''' 
    Interface for Exploration strategies.

    Requires method for plotting the Exploration schedule implied by the hyperparameters,
    as well as method for choosing the next move.  
    '''
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
    
    @abstractmethod
    def plot(self):
        ''' Plot to be called before training to see implied exploration curve given the parameters. '''
    
    @property
    @abstractmethod
    def strategy_name(self):
        ''' Also require property: strategy name, to be saved in MLflow and also used in the train loop '''

class EpsilonGreedyStrategy(ExplorationStrategy):
    ''' 
    Implementation of Epsilon Greedy Exploration 

    Makes a move at random with p = eps_threshold and takes the move with the highest predicted Q-value with p = 1-eps_threshold.
    '''
    def __init__(self, config):
        super().__init__(config)
        assert config.eps_decay, 'Provide eps_decay in EpsilonGreedyStrategy.'
        assert config.eps_start, 'Provide eps_start in EpsilonGreedyStrategy.'
        assert config.eps_end, 'Provide eps_end in EpsilonGreedyStrategy.'

    @property
    def eps_threshold(self):
        ''' Calculates the chance of taking a random move at a given time '''
        p_random = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * math.exp(-1. * self.n_steps / self.config.eps_decay)
        if self.n_steps % 1000 == 0:
            print(f'Step {self.n_steps}, Chance of random move: {p_random}')
        return p_random
    
    @property
    def strategy_name(self):
        return 'Epsilon Greedy'
    
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
        return None

    def plot(self):
        ''' Plot the epsilon decay schedule over steps taken '''
        steps = range(self.config.eps_decay * 5)
        epsilons = [
            self.config.eps_end + (self.config.eps_start - self.config.eps_end) * math.exp(-1. * step / self.config.eps_decay)
            for step in steps
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, epsilons)
        plt.xlabel('Steps')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Time')
        plt.grid(True)
        plt.show()

class BoltzmannExploration(ExplorationStrategy):
    ''' Implementation of Epsilon Greedy Move '''

    def __init__(self, config):
        super().__init__(config)
        assert config.tau0, 'Provide tau0 in BoltzmannExploration.'
        assert config.taumin, 'Provide taumin in BoltzmannExploration.'
        assert config.tau_decay_rate, 'Provide tau_decay_rate in BoltzmannExploration.'

    @property
    def tau(self):
        ''' temperature parameter for boltzmann exploration depending on time. 1 means spread out and smaller will peak at the best choice '''
        taut = self.config.tau0 * math.exp(-self.config.tau_decay_rate * self.n_steps)
        taut = max(taut, self.config.taumin)
        if self.n_steps % 1000 == 0:
            print(f'Step {self.n_steps}, temperature: {taut}')
        return taut
    
    @property
    def strategy_name(self):
        return 'Boltzmann Exploration'
    
    @increment_steps
    def choose_next_move(
                        self,
                        predictions,
                        possible_moves
                        ):
        if possible_moves:
            idxs = [move[1] for move in possible_moves]
            arr = np.array([predictions[idx] for idx in idxs])
            preds_rescaled = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            weights = np.exp(preds_rescaled/self.tau)
            weights = weights/np.sum(weights)
            return random.choices(
                possible_moves,
                weights=weights,
                k=1
                )[0]
        else:
            rand_move = random.choice(possible_moves)
            return rand_move
    
    def plot(self):
        ''' Plot the epsilon decay curve over time '''
        steps = range(self.config.eps_decay * 5)  # Arbitrary range long enough to see decay
        epsilons = [
            self.config.tau0 * math.exp(-self.config.tau_decay_rate * step)
            for step in steps
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, epsilons)
        plt.xlabel('Steps')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Time')
        plt.grid(True)
        plt.show()