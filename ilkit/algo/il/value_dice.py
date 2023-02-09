from typing import Dict, Union

import numpy as np
import torch as th

from ilkit.algo.base import ILPolicy
from ilkit.util.logger import BaseLogger


class ValueDICE(ILPolicy):
    """Imitation Learning via Off-Policy Distribution Matching (ValueDICE)
    """

    def __init__(self, cfg: Dict, logger: BaseLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        ...

    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg
    ) -> Union[np.ndarray, th.Tensor]:
        ...

    def update(self):
        ...
