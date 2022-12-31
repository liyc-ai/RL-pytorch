from typing import Dict

from ilkit.algo.il import ILPolicy


class InfoGAIL(ILPolicy):
    """Interpretable Imitation Learning from Visual Demonstrations (InfoGAIL)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
