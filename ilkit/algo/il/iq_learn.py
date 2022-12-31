from typing import Dict

from ilkit.algo.il import ILPolicy


class IQLearnContinuous(ILPolicy):
    """Inverse soft-Q Learning for Imitation (IQ-Learn), continuous action space
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)


class IQLearnDiscrete(ILPolicy):
    """Inverse soft-Q Learning for Imitation (IQ-Learn), discrete action space
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
