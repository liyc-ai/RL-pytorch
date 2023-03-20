from ilkit.algo.base import BasePolicy
from ilkit.trainer.base import BaseTrainer

# imitation learning


class BCTrainer(BaseTrainer):
    def __init__(self, policy: BasePolicy):
        super().__init__(policy)

    def train(self):
        ...


# reinforcement learning
