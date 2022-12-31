import random
from typing import Dict

import torch as th
from torch import optim
from torch.utils.data import BatchSampler

from ilkit.algo.rl.trpo import TRPO
from ilkit.util.ptu import gradient_descent


class PPO(TRPO):
    """Proximal Policy Optimization (PPO)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_param(self):
        # hyper-param
        self.lambda_ = self.algo_config["lambda_"]
        self.epsilon = self.algo_config["epsilon"]

    def init_component(self):
        super().init_component()

        self.actor_optim = getattr(optim, self.algo_config["actor"]["optimizer"])(
            self.actor.parameters(), self.algo_config["actor"]["lr"]
        )

        self.models.update({"actor_optim": self.actor_optim})

    def _update_actor(self, states: th.Tensor, actions: th.Tensor):
        with th.no_grad():
            _, old_log_probs = self._get_action_dist(states, actions)

        idx = list(range(self.trans_buffer.size))
        for _ in range(self.algo_config["value_net"]["n_update"]):
            random.shuffle(idx)
            batches = list(
                BatchSampler(idx, batch_size=self.batch_size, drop_last=False)
            )
            for batch in batches:
                sampled_states, sampled_actions = states[batch], actions[batch]
                sampled_action_dist, sampled_log_probs = self._get_action_dist(
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
                    - self.algo_config["entropy_coef"] * sampled_action_dist.entropy().mean()
                )
                self.log_info.update(
                    {
                        "loss/actor": gradient_descent(
                            self.actor_optim,
                            loss,
                            self.actor.parameters(),
                            # experimental results show that clipping grad realy improves performance
                            self.algo_config["actor"]["clip"],
                        )
                    }
                )
