import torch as th
from omegaconf import DictConfig
from rlplugs.logger import LoggerType

from rlpyt.dqn import DQNAgent


class DDQNAgent(DQNAgent):
    """Deep Double Q Networks (DDQN)"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _get_q_target(self, next_states: th.Tensor):
        with th.no_grad():
            _next_action = th.argmax(self.q_net(next_states), -1, True)
            q_target = th.gather(self.q_net_target(next_states), -1, _next_action)
        return q_target
