from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import tempfile
import os


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
    epsilon0: Optional[float] = 0.5
    epsilonmin: Optional[float] = 0.05
    epsilon_decay_rate: Optional[float] = 150000
    tau0: Optional[float] = 0.5
    taumin: Optional[float] = 0.01
    tau_decay_rate: Optional[float] = 0.0001
class IExplorationStrategy(Protocol):
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

    def choose_next_move(
        self,
        predictions,
        possible_moves
        ):
        ''' Stratgegy to decide on the next move '''

    def plot(self):
        ''' Plot to be called before training to see implied exploratteion curve given the parameters. '''

class EpsilonGreedyStrategy(IExplorationStrategy):
    ''' 
    Implementation of Epsilon Greedy Exploration 

    Makes a move at random with p = eps_threshold and takes the move with the highest predicted Q-value with p = 1-eps_threshold.
    '''
    def __init__(self, config):
        self.config = config
        self.n_steps = 0
        assert config.epsilon0, 'Provide epsilon0 for HybridExploration.'
        assert config.epsilonmin, 'Provide epsilonmin for HybridExploration.'
        assert config.epsilon_decay_rate, 'Provide tau_decay_rate for HybridExploration.'

    @property
    def eps(self):
        ''' temperature parameter for epsilon exploration depending on time.'''
        epst = self.config.epsilonmin + (self.config.epsilon0 - self.config.epsilonmin) * math.exp(-1. * self.n_steps / self.config.epsilon_decay_rate)
        if self.n_steps % 1000 == 0:
            print(f'Step {self.n_steps}, temperature: {epst}')
        return epst
      
    def choose_next_move(
                        self,
                        predictions,
                        possible_moves
                        ) -> int:
        assert len(possible_moves) > 0, "no move possible -> bug in game setup"
        self.n_steps += 1
        if random.random() < self.eps:
            return random.choice(possible_moves)
        return max(possible_moves, key=lambda idx: predictions[idx])

    def plot(self):
        ''' Plot the epsilon decay schedule over steps taken '''
        steps = range(0, 4_000_000, 1_000)
        epsilons = [
            self.config.epsilonmin + (self.config.epsilon0 - self.config.epsilonmin) * math.exp(-1. * step / self.config.epsilon_decay_rate)
            for step in steps
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, epsilons)
        plt.xlabel('Steps')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Time')
        plt.grid(True)
        plt.savefig("visualisations/exploration_schedule.png")


class BoltzmannExploration(IExplorationStrategy):
    """ Implementation of Boltzmann(Softmax) Exploration """

    def __init__(self, config):
        self.config = config
        self.n_steps = 0
        assert config.tau0, "Provide tau0 in BoltzmannExploration."
        assert config.taumin, "Provide taumin in BoltzmannExploration."
        assert config.tau_decay_rate, "Provide tau_decay_rate in BoltzmannExploration."

    @property
    def tau(self):
        """ temperature parameter for boltzmann exploration depending on time. 1 means spread out and smaller will peak at the best choice """
        taut = self._calc_taut(self.n_steps)
        if self.n_steps % 1000 == 0:
            print(f'Step {self.n_steps}, temperature: {taut}')
        return taut

    def _calc_taut(self, step):
        taut = self.config.tau0 * math.exp(-self.config.tau_decay_rate * step)
        taut = max(taut, self.config.taumin)
        return taut
    
    def choose_next_move(
                        self,
                        predictions,
                        possible_moves
                        ) -> int:
        assert len(possible_moves) > 0, "no move possible -> bug in game setup"
        self.n_steps += 1
        arr = np.array([predictions[idx] for idx in possible_moves])
        preds_rescaled = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        weights = np.exp(preds_rescaled/self.tau)
        weights /= weights.sum()
        return random.choices(
            possible_moves,
            weights=weights,
            k=1
            )[0]

    
    def plot(self):
        ''' Plot the exploration curve over time '''
        steps = range(0, 4_000_000, 1_000)
        taus = [
            self._calc_taut(step)
            for step in steps
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, taus)
        plt.xlabel('Steps')
        plt.ylabel('Tau')
        plt.title('Tau vs iterations')
        plt.grid(True)
        plt.show()
        plt.savefig("visualisations/exploration_schedule.png")


class HybridExploration(IExplorationStrategy):
    ''' Hybrid Exploration Strategy, using mix of Boltzmann and Epsilon-Greedy '''

    def __init__(self, config):
        self.config = config
        self.n_steps = 0
        assert config.tau0, 'Provide tau0 for HybridExploration.'
        assert config.taumin, 'Provide taumin for HybridExploration.'
        assert config.tau_decay_rate, 'Provide tau_decay_rate for HybridExploration.'
        assert config.epsilon0, 'Provide epsilon0 for HybridExploration.'
        assert config.epsilonmin, 'Provide epsilonmin for HybridExploration.'
        assert config.epsilon_decay_rate, 'Provide tau_decay_rate for HybridExploration.'

    @property
    def tau(self):
        ''' temperature parameter for boltzmann exploration depending on time. 1 means spread out and smaller will peak at the best choice '''
        taut = self.config.tau0 * math.exp(-self.config.tau_decay_rate * self.n_steps)
        taut = max(taut, self.config.taumin)
        if self.n_steps % 1000 == 0:
            print(f'Step {self.n_steps}, temperature: {taut}')
        return taut

    @property
    def epsilon(self):
        ''' temperature parameter for epsilon exploration depending on time.'''
        epst = self.config.epsilon0 * math.exp(-self.config.epsilon_decay_rate * self.n_steps) + self.config.epsilonmin
        if self.n_steps % 1000 == 0:
            print(f'Step {self.n_steps}, temperature: {epst}')
        return epst

    def choose_next_move(
                        self,
                        predictions,
                        possible_moves
                        ):
        assert len(possible_moves) > 0, "no move possible -> bug in game setup"
        self.n_steps += 1
        idxs = [move[1] for move in possible_moves]
        
        if random.random():
            return random.choices(possible_moves)[0]
        
        arr = np.array([predictions[idx] for idx in idxs])
        preds_rescaled = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        
        weights = np.exp(preds_rescaled/self.tau)
        weights = weights/np.sum(weights)
        
        return random.choices(
            possible_moves,
            weights=weights,
            k=1
            )[0]

    
    def plot(self):
        ''' Plot the exploration curve over time '''
        steps = range(0, 4_000_000, 1_000)
        taus = [
            self.config.tau0 * math.exp(-self.config.tau_decay_rate * step)
            for step in steps
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, taus)
        plt.xlabel('Steps')
        plt.ylabel('Tau')
        plt.title('Tau vs iterations')
        plt.grid(True)
        plt.savefig("visualisations/exploration_schedule.png")
