from typing import Dict

import torch as th
from rlplugs.logger import LoggerType

from rlpyt.algo.dqn import DQN


class DDQN(DQN):
    """Deep Double Q Networks (DDQN)"""

    def __init__(self, cfg: Dict, logger: LoggerType):
        super().__init__(cfg, logger)

    def _get_q_target(self, next_states: th.Tensor):
        with th.no_grad():
            _next_action = th.argmax(self.q_net(next_states), -1, True)
            q_target = th.gather(self.q_net_target(next_states), -1, _next_action)
        return q_target
