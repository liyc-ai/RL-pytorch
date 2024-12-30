from copy import deepcopy
from typing import Dict, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from exputils.net.actor import MLPDeterministicActor
from exputils.net.critic import MLPTwinCritic
from exputils.net.ptu import freeze_net, gradient_descent, move_device, polyak_update
from omegaconf import DictConfig
from torch import nn, optim

from .base import BaseRLAgent


class TD3Agent(BaseRLAgent):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3)"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def setup_model(self):
        # hyper-param
        self.warmup_steps = self.cfg.agent.warmup_steps
        self.env_steps = self.cfg.agent.env_steps
        self.total_train_it = 0

        # actor
        actor_kwarg = {
            "state_shape": self.state_shape,
            "net_arch": self.cfg.agent.actor.net_arch,
            "action_shape": self.action_shape,
            "activation_fn": getattr(nn, self.cfg.agent.actor.activation_fn),
        }
        self.actor = MLPDeterministicActor(**actor_kwarg)
        self.actor_target = deepcopy(self.actor)
        self.actor_optim = getattr(optim, self.cfg.agent.actor.optimizer)(
            self.actor.parameters(), self.cfg.agent.actor.lr
        )

        # critic
        critic_kwarg = {
            "input_shape": (self.state_shape[0] + self.action_shape[0],),
            "net_arch": self.cfg.agent.critic.net_arch,
            "output_shape": (1,),
            "activation_fn": getattr(nn, self.cfg.agent.critic.activation_fn),
        }
        self.critic = MLPTwinCritic(**critic_kwarg)
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = getattr(optim, self.cfg.agent.critic.optimizer)(
            self.critic.parameters(), self.cfg.agent.critic.lr
        )

        freeze_net((self.actor_target, self.critic_target))
        move_device(
            (self.actor, self.actor_target, self.critic, self.critic_target),
            self.device,
        )

        self.models.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "actor_optim": self.actor_optim,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "critic_optim": self.critic_optim,
            }
        )

    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        actor: nn.Module = None,
        **kwargs,
    ) -> th.Tensor:
        state = th.Tensor(state).to(self.device) if type(state) is np.ndarray else state

        if actor is None:
            action = self.actor(state)
        else:
            action = actor(state)

        action = th.tanh(action)

        # add explore noise
        if not deterministic:
            noise = th.clamp(
                th.randn_like(action) * self.cfg.agent.sigma,
                -self.cfg.agent.c,
                self.cfg.agent.c,
            )
            action = th.clamp(action + noise, -1.0, 1.0)

        return action

    def update(self) -> Dict:
        self.stats = dict()
        rest_steps = self.trans_buffer.size - self.warmup_steps
        if not (
            self.trans_buffer.size < self.batch_size
            or rest_steps < 0
            or rest_steps % self.env_steps != 0
        ):
            self.total_train_it += 1
            states, actions, next_states, rewards, dones = self.trans_buffer.sample(
                self.batch_size
            )

            # update params
            for _ in range(self.env_steps):
                self._update_critic(states, actions, next_states, rewards, dones)
                if self.total_train_it % self.cfg.agent.policy_freq == 0:
                    self._update_actor(states)

                    polyak_update(
                        self.critic.parameters(),
                        self.critic_target.parameters(),
                        self.cfg.agent.critic.tau,
                    )
                    polyak_update(
                        self.actor.parameters(),
                        self.actor_target.parameters(),
                        self.cfg.agent.actor.tau,
                    )

        return self.stats

    def _update_critic(
        self,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
        rewards: th.Tensor,
        dones: th.Tensor,
    ):
        with th.no_grad():
            pred_next_actions = self.select_action(
                next_states,
                deterministic=False,
                actor=self.actor_target,
            )
            target_Q1, target_Q2 = self.critic_target(
                True, next_states, pred_next_actions
            )
            target_Q = rewards + self.gamma * (1 - dones) * th.min(target_Q1, target_Q2)
        Q1, Q2 = self.critic(True, states, actions)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self.stats.update(
            {"loss/critic": gradient_descent(self.critic_optim, critic_loss)}
        )

    def _update_actor(self, states: th.Tensor):
        pred_actions = self.select_action(states, deterministic=True)
        Q = self.critic(False, states, pred_actions)
        actor_loss = -th.mean(Q)
        self.stats.update(
            {"loss/actor": gradient_descent(self.actor_optim, actor_loss)}
        )
