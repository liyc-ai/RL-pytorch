import random
from typing import Dict

import torch as th
from rlplugs.drls.gae import GAE
from rlplugs.logger import LoggerType
from rlplugs.net.actor import MLPGaussianActor
from rlplugs.net.critic import MLPCritic
from rlplugs.net.ptu import gradient_descent, move_device
from torch import nn, optim
from torch.utils.data import BatchSampler

from rlpyt.trpo import TRPO


class PPO(TRPO):
    """Proximal Policy Optimization (PPO)"""

    def __init__(self, cfg: Dict, logger: LoggerType):
        super().__init__(cfg, logger)

    def setup_model(self):
        # hyper-param
        self.epsilon = self.algo_cfg["epsilon"]
        self.lambda_ = self.algo_cfg["lambda_"]

        # GAE
        self.gae = GAE(
            self.gamma,
            self.lambda_,
            self.algo_cfg["norm_adv"],
            self.algo_cfg["use_td_lambda"],
        )

        # actor
        actor_kwarg = {
            "state_shape": self.state_shape,
            "net_arch": self.algo_cfg["actor"]["net_arch"],
            "action_shape": self.action_shape,
            "activation_fn": getattr(nn, self.algo_cfg["actor"]["activation_fn"]),
        }
        self.actor = MLPGaussianActor(**actor_kwarg)
        self.actor_optim = getattr(optim, self.algo_cfg["actor"]["optimizer"])(
            self.actor.parameters(), self.algo_cfg["actor"]["lr"]
        )

        # value network
        value_net_kwarg = {
            "input_shape": self.state_shape,
            "output_shape": (1,),
            "net_arch": self.algo_cfg["value_net"]["net_arch"],
            "activation_fn": getattr(nn, self.algo_cfg["value_net"]["activation_fn"]),
        }
        self.value_net = MLPCritic(**value_net_kwarg)
        self.value_net_optim = getattr(optim, self.algo_cfg["value_net"]["optimizer"])(
            self.value_net.parameters(), self.algo_cfg["value_net"]["lr"]
        )

        move_device((self.actor, self.value_net), self.device)

        self.models.update(
            {
                "actor": self.actor,
                "actor_optim": self.actor_optim,
                "value_net": self.value_net,
                "value_net_optim": self.value_net_optim,
            }
        )

    def _update_actor(self, states: th.Tensor, actions: th.Tensor):
        with th.no_grad():
            _, old_log_probs = self._select_action_dist(states, actions)

        idx = list(range(self.trans_buffer.size))
        for _ in range(self.algo_cfg["value_net"]["n_update"]):
            random.shuffle(idx)
            batches = list(
                BatchSampler(idx, batch_size=self.batch_size, drop_last=False)
            )
            for batch in batches:
                sampled_states, sampled_actions = states[batch], actions[batch]
                sampled_action_dist, sampled_log_probs = self._select_action_dist(
                    sampled_states, sampled_actions
                )
                ratio = th.exp(sampled_log_probs - old_log_probs[batch])
                surr1 = ratio * self.adv[batch]
                surr2 = (
                    th.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * self.adv[batch]
                )
                loss = (
                    -th.min(surr1, surr2).mean()
                    - self.algo_cfg["entropy_coef"]
                    * sampled_action_dist.entropy().mean()
                )
                self.log_info.update(
                    {
                        "loss/actor": gradient_descent(
                            self.actor_optim,
                            loss,
                            self.actor.parameters(),
                            # experimental results show that clipping grad realy improves performance
                            self.algo_cfg["actor"]["clip"],
                        )
                    }
                )
