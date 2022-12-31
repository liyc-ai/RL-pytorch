from typing import Dict

from ilkit.algo.il import ILPolicy


class DAC(ILPolicy):
    """Discriminator-Actor-Critic (DAC)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
