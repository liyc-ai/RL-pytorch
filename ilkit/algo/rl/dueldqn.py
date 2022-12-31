from copy import deepcopy
from typing import Dict

from torch import nn, optim

from ilkit.algo.rl.ddqn import DDQN
from ilkit.net.critic import MLPDuleQNet
from ilkit.util.ptu import freeze_net, move_device


class DuelDQN(DDQN):
    """Dueling Deep Q Networks (DuelDQN)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_component(self):
        # Q network
        q_net_kwarg = {
            "input_shape": self.state_shape,
            "output_shape": self.action_shape,
            "net_arch": self.algo_config["QNet"]["net_arch"],
            "v_head": self.algo_config["QNet"]["v_head"],
            "adv_head": self.algo_config["QNet"]["adv_head"],
            "activation_fn": getattr(nn, self.algo_config["QNet"]["activation_fn"]),
            "mix_type": self.algo_config["QNet"]["mix_type"],
        }
        self.q_net = MLPDuleQNet(**q_net_kwarg)
        self.q_net_target = deepcopy(self.q_net)
        self.q_net_optim = getattr(optim, self.algo_config["QNet"]["optimizer"])(
            self.q_net.parameters(), self.algo_config["QNet"]["lr"]
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
