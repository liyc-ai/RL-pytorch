from typing import Dict

from ilkit.algo.il import ILPolicy


class ValueDICE(ILPolicy):
    """Imitation Learning via Off-Policy Distribution Matching (ValueDICE)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
