from copy import deepcopy

from drlplugs.net.critic import MLPDuleQNet
from drlplugs.net.ptu import freeze_net, move_device
from omegaconf import DictConfig
from torch import nn, optim

from rlpyt.ddqn import DDQNAgent


class DuelDQNAgent(DDQNAgent):
    """Dueling Deep Q Networks (DuelDQN)"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def setup_model(self):
        # hyper-param
        self.target_update_freq = self.cfg.agent.target_update_freq
        self.epsilon = self.cfg.agent.epsilon
        self.global_t = 0

        # Q network
        q_net_kwarg = {
            "input_shape": self.state_shape,
            "output_shape": self.action_shape,
            "net_arch": self.cfg.agent.QNet.net_arch,
            "v_head": self.cfg.agent.QNet.v_head,
            "adv_head": self.cfg.agent.QNet.adv_head,
            "activation_fn": getattr(nn, self.cfg.agent.QNet.activation_fn),
            "mix_type": self.cfg.agent.QNet.mix_type,
        }
        self.q_net = MLPDuleQNet(**q_net_kwarg)
        self.q_net_target = deepcopy(self.q_net)
        self.q_net_optim = getattr(optim, self.cfg.agent.QNet.optimizer)(
            self.q_net.parameters(), self.cfg.agent.QNet.lr
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
