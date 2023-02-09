import random
from itertools import chain
from typing import Dict

import torch as th
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import BatchSampler

from ilkit.algo.il.gail import GAIL
from ilkit.net.critic import MLPCritic
from ilkit.util.logger import BaseLogger
from ilkit.util.ptu import gradient_descent, move_device


class AIRL(GAIL):
    """Learning Robust Rewards with Adversarial Inverse Reinforcement Learning (AIRL)
    """

    def __init__(self, cfg: Dict, logger: BaseLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        super().setup_model()

        # hyper-param
        self.gamma = self.algo_cfg["gamma"]

    def _init_disc(self):
        g_kwarg = {
            "input_shape": (self.state_shape[0] + self.action_shape[0],),
            "output_shape": (1,),
            "net_arch": self.algo_cfg["discriminator"]["g"]["net_arch"],
            "activation_fn": getattr(
                nn, self.algo_cfg["discriminator"]["g"]["activation_fn"]
            ),
        }
        self.g = MLPCritic(**g_kwarg)

        h_kwarg = {
            "input_shape": self.state_shape,
            "output_shape": (1,),
            "net_arch": self.algo_cfg["discriminator"]["h"]["net_arch"],
            "activation_fn": getattr(
                nn, self.algo_cfg["discriminator"]["h"]["activation_fn"]
            ),
        }
        self.h = MLPCritic(**h_kwarg)

        move_device((self.g, self.h), self.device)

        self.disc_optim = getattr(optim, self.algo_cfg["discriminator"]["optimizer"])(
            chain(self.g.parameters(), self.h.parameters()),
            self.algo_cfg["discriminator"]["lr"],
        )

        self.models.update({"g": self.g, "h": self.h, "disc_optim": self.disc_optim})

    def disc(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        f = (
            self.g(state, action)
            + self.gamma * (1 - done) * self.h(next_state)
            - self.h(state)
        )
        with th.no_grad():
            _, log_prob = self.generator._select_action_dist(state, action)
        return f - log_prob

    def _update_disc(self):
        for _ in range(self.algo_cfg["discriminator"]["n_update"]):
            random.shuffle(self.idx)
            batches = list(
                BatchSampler(self.idx, batch_size=self.batch_size, drop_last=False)
            )
            for batch in batches:
                # expert
                expert_states, expert_actions, expert_next_states, _, expert_dones = self.expert_buffer.sample(
                    self.batch_size
                )
                d_expert = self.disc(
                    expert_states, expert_actions, expert_next_states, expert_dones
                )
                expert_loss = F.binary_cross_entropy_with_logits(
                    d_expert, th.ones_like(d_expert)
                )

                # imitator
                imitator_states, imitator_actions, imitator_next_states, _, imitator_dones = [
                    buffer[batch] for buffer in self.generator.trans_buffer.buffers
                ]
                d_imitator = self.disc(
                    imitator_states,
                    imitator_actions,
                    imitator_next_states,
                    imitator_dones,
                )
                imitator_loss = F.binary_cross_entropy_with_logits(
                    d_imitator, th.zeros_like(d_imitator)
                )

                loss = expert_loss + imitator_loss
                self.log_info.update(
                    {"loss/disc": gradient_descent(self.disc_optim, loss)}
                )

    def _prepare_reward(self):
        with th.no_grad():
            states, actions, next_states, _, dones = self.generator.trans_buffer.buffers
            # - log(1-D(s,a)), trick from GAN
            rewards = -F.logsigmoid(-self.disc(states, actions, next_states, dones))
            self.generator.trans_buffer.buffers[3] = rewards
