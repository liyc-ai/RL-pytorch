import math
from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn, optim

from src.utils.net.actor import MLPGaussianActor
from src.utils.net.critic import MLPTwinCritic
from src.utils.net.ptu import freeze_net, gradient_descent, move_device, polyak_update

from .base import BaseRLAgent


class SACAgent(BaseRLAgent):
    """Soft Actor Critic (SAC)"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def setup_model(self):
        # hyper-param
        self.entropy_target = -self.action_shape[0]
        self.warmup_steps = self.cfg.agent.warmup_steps
        self.env_steps = self.cfg.agent.env_steps

        # actor
        actor_kwarg = {
            "state_shape": self.state_shape,
            "net_arch": self.cfg.agent.actor.net_arch,
            "action_shape": self.action_shape,
            "state_std_independent": self.cfg.agent.actor.state_std_independent,
            "activation_fn": getattr(nn, self.cfg.agent.actor.activation_fn),
        }
        self.actor = th.compile(MLPGaussianActor(**actor_kwarg))
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
        self.critic = th.compile(MLPTwinCritic(**critic_kwarg))
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = getattr(optim, self.cfg.agent.critic.optimizer)(
            self.critic.parameters(), self.cfg.agent.critic.lr
        )

        # alpha, we optimize log(alpha) because alpha should always be bigger than 0.
        if self.cfg.agent.log_alpha.auto_tune:
            self.log_alpha = th.tensor(
                [self.cfg.agent.log_alpha.init_value],
                device=self.device,
                requires_grad=True,
            )
            self.log_alpha_optim = getattr(optim, self.cfg.agent.log_alpha.optimizer)(
                [self.log_alpha], self.cfg.agent.log_alpha.lr
            )
            self.models.update(
                {"log_alpha": self.log_alpha, "log_alpha_optim": self.log_alpha_optim}
            )
        else:
            self.log_alpha = th.tensor(
                [self.cfg.agent.log_alpha.init_value], device=self.device
            )

        freeze_net((self.critic_target,))
        move_device((self.actor, self.critic, self.critic_target), self.device)

        self.models.update(
            {
                "actor": self.actor,
                "actor_optim": self.actor_optim,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "critic_optim": self.critic_optim,
            }
        )

    @property
    def alpha(self):
        return math.exp(self.log_alpha.item())

    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        return_log_prob: bool,
        **kwarg,
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        :param deterministic: whether sample from the action distribution or just the action mean.
        :param return_dtype_tensor: whether the returned data's dtype keeps to be torch.Tensor or numpy.ndarray
        :param return_log_prob: whether return log_prob
        """
        # Due to the squash operation, we need that keep_dtype_tensor == True here.
        if return_log_prob:
            action, log_prob = self.actor.sample(
                state, deterministic, True, self.device
            )
            # squash action
            log_prob -= th.sum(
                2 * (np.log(2.0) - action - F.softplus(-2 * action)),
                axis=-1,
                keepdims=True,
            )
        else:
            action = self.actor.sample(state, deterministic, False, self.device)
            log_prob = None

        action = th.tanh(action)

        # # scale, we could instead use gym.wrappers to rescale action space
        # # [-1, +1] -> [-action_scale, action_scale]
        # if return_log_prob:
        #     log_prob -= th.sum(
        #         np.log(1.0 / self.action_scale) * th.ones_like(action),
        #         axis=-1,
        #         keepdim=True,
        #     )
        # action *= self.action_scale

        return (action, log_prob) if return_log_prob else action

    def update(self) -> Dict:
        self.stats = dict()
        rest_steps = self.trans_buffer.size - self.warmup_steps
        if not (
            self.trans_buffer.size < self.batch_size
            or rest_steps < 0
            or rest_steps % self.env_steps != 0
        ):
            states, actions, next_states, rewards, dones = self.trans_buffer.sample(
                self.batch_size
            )

            # update params
            for _ in range(self.env_steps):
                self._update_critic(states, actions, next_states, rewards, dones)
                if self.cfg.agent.log_alpha.auto_tune:
                    self._update_alpha(self._update_actor(states))
                else:
                    self._update_actor(states)
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.cfg.agent.critic.tau,
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
            pred_next_actions, pred_next_log_pis = self.select_action(
                next_states,
                deterministic=False,
                return_log_prob=True,
            )
            target_Q1, target_Q2 = self.critic_target(
                True, next_states, pred_next_actions
            )
            target_Q = th.min(target_Q1, target_Q2) - self.alpha * pred_next_log_pis
            target_Q = rewards + self.gamma * (1 - dones) * target_Q

        Q1, Q2 = self.critic(True, states, actions)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.stats.update(
            {"loss/critic": gradient_descent(self.critic_optim, critic_loss)}
        )

    def _update_actor(self, states: th.Tensor):
        pred_actions, pred_log_pis = self.select_action(
            states, deterministic=False, return_log_prob=True
        )
        Q1, Q2 = self.critic(True, states, pred_actions)
        actor_loss = th.mean(self.alpha * pred_log_pis - th.min(Q1, Q2))
        self.stats.update(
            {"loss/actor": gradient_descent(self.actor_optim, actor_loss)}
        )

        return pred_log_pis.detach()

    def _update_alpha(self, pred_log_pis: th.Tensor):
        """Auto-tune alpha

        Note: pred_log_pis are detached from the computation graph
        """
        alpha_loss = th.mean(self.log_alpha * (-pred_log_pis - self.entropy_target))
        self.stats.update(
            {"loss/alpha": gradient_descent(self.log_alpha_optim, alpha_loss)}
        )

        # update alpha
        self.models["log_alpha"].data = self.log_alpha.data
        self.models["log_alpha"].data = self.log_alpha.data
