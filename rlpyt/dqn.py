from copy import deepcopy
from typing import Dict, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from rlplugs.logger import LoggerType
from rlplugs.net.critic import MLPCritic
from rlplugs.net.ptu import freeze_net, gradient_descent, move_device
from torch import nn, optim

from rlpyt import OnlineRLAgent


class DQN(OnlineRLAgent):
    """Deep Q Networks (DQN)"""

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
            "activation_fn": getattr(nn, self.algo_cfg["QNet"]["activation_fn"]),
        }
        self.q_net = MLPCritic(**q_net_kwarg)
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

    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        **kwarg
    ) -> Union[th.Tensor, np.ndarray]:
        if not deterministic and np.random.random() < self.epsilon:
            return kwarg["action_space"].sample()
        with th.no_grad():
            state = (
                th.Tensor(state).to(self.device) if type(state) is np.ndarray else state
            )
            pred_q = self.q_net(state)
            action = th.argmax(pred_q, dim=-1)
        if keep_dtype_tensor:
            return action
        else:
            return action.cpu().numpy()

    def _get_q_target(self, next_states: th.Tensor):
        with th.no_grad():
            q_target, _ = th.max(self.q_net_target(next_states), -1, True)
        return q_target

    def _get_q(self, states: th.Tensor, actions: th.Tensor):
        q = th.gather(self.q_net(states), -1, actions)
        return q

    def update(self) -> Dict:
        self.log_info = dict()
        if self.trans_buffer.size >= self.batch_size:
            self.global_t += 1
            states, actions, next_states, rewards, dones = self.trans_buffer.sample(
                self.batch_size, shuffle=True
            )

            # calculate q target and td target
            q_target = self._get_q_target(next_states)
            td_target = rewards + self.gamma * (1 - dones) * q_target

            # calculate q
            q = self._get_q(states, actions)

            # update q network
            loss = F.mse_loss(q, td_target)
            self.log_info.update(
                {
                    "loss": gradient_descent(self.q_net_optim, loss),
                    "Q/q": th.mean(q).item(),
                    "Q/q_target": th.mean(q_target).item(),
                }
            )

            # update q target
            if self.global_t % self.target_update_freq == 0:
                self.q_net_target.load_state_dict(self.q_net.state_dict())

        return self.log_info
