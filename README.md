# Using Reinforcement Learning on the Board Game Take It Easy!


## Overview

### Reinforcement Learning frameworks used
This project uses Deep Double Q-Learning to train an Agent by playing a large number of games against itself, with no further guidance provided.

The game is encoded using different variations of one-hot encoding for the boardstates.

Further Techniques used:
* Prioritized Experience Replay (PER)
* Boltzmann-Exploration
* Reward-Shaping

### Design Principles

The Software relies heavily on Composition for three reasons:
* easy changes for parts without introducing errors in others
* easy to toggle on/off features to study impact
* very useful for tracking Experiments using mlflow

There is also a clear seperation between the game itself and the agent. The Implementation relies heavily on Python classes and therefor focusses on convenience and readibility and not on performance.
Deep-Learning is implemented in Pytorch.
