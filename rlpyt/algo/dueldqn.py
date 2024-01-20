from copy import deepcopy
from typing import Dict

from rlplugs.logger import LoggerType
from torch import nn, optim

from rlpyt.algo.ddqn import DDQN
from rlpyt.net.critic import MLPDuleQNet
from rlpyt.util.ptu import freeze_net, move_device


class DuelDQN(DDQN):
    """Dueling Deep Q Networks (DuelDQN)"""

    def __init__(self, cfg: Dict, logger: LoggerType):
        super().__init__(cfg, logger)

    def setup_model(self):
        # hyper-param
        self.target_update_freq = self.algo_cfg["target_update_freq"]
        self.epsilon = self.algo_cfg["epsilon"]
        self.global_t = 0

        # Q network
        q_net_kwarg = {
            "input_shape": self.state_shape,
            "output_shape": self.action_shape,
            "net_arch": self.algo_cfg["QNet"]["net_arch"],
            "v_head": self.algo_cfg["QNet"]["v_head"],
            "adv_head": self.algo_cfg["QNet"]["adv_head"],
            "activation_fn": getattr(nn, self.algo_cfg["QNet"]["activation_fn"]),
            "mix_type": self.algo_cfg["QNet"]["mix_type"],
        }
        self.q_net = MLPDuleQNet(**q_net_kwarg)
        self.q_net_target = deepcopy(self.q_net)
        self.q_net_optim = getattr(optim, self.algo_cfg["QNet"]["optimizer"])(
            self.q_net.parameters(), self.algo_cfg["QNet"]["lr"]
        )

        freeze_net((self.q_net_target,))
        move_device((self.q_net, self.q_net_target), self.device)

        self.models.update(
            {
                "q_net": self.q_net,
                "q_net_target": self.q_net_target,
                "q_net_optim": self.q_net_optim,
            }
        )
