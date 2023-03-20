from abc import ABC, abstractmethod

from algo.base import BasePolicy


# imitation learning
class AILTrainer:
    def __init__(self, policy: BasePolicy):
        super().__init__(policy)

    def train(self):
        ...


# reinforcement learning
