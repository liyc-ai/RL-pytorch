from abc import ABC, abstractmethod

from algo.base import BasePolicy


class BaseTrainer(ABC):
    """Interface for trainers
    """

    def __init__(self, policy: BasePolicy):
        self.policy = policy

    @abstractmethod
    def train(self):
        raise NotImplementedError
