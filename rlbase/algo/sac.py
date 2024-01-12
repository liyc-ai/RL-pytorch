import math
from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from mllogger import TBLogger
from stable_baselines3.common.utils import polyak_update
from torch import nn, optim

from rlbase.algo.base import OnlineRLPolicy
from rlbase.net.actor import MLPGaussianActor
from rlbase.net.critic import MLPTwinCritic
from rlbase.util.ptu import (freeze_net, gradient_descent, move_device,
                            tensor2ndarray)


class SAC(OnlineRLPolicy):
    """Soft Actor Critic (SAC)
    """

    def __init__(self, cfg: Dict, logger: TBLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        # hyper-param
        self.entropy_target = -self.action_shape[0]
        self.warmup_steps = self.algo_cfg["warmup_steps"]
        self.env_steps = self.algo_cfg["env_steps"]

        # actor
        actor_kwarg = {
            "state_shape": self.state_shape,
            "net_arch": self.algo_cfg["actor"]["net_arch"],
            "action_shape": self.action_shape,
            "state_std_independent": self.algo_cfg["actor"]["state_std_independent"],
            "activation_fn": getattr(nn, self.algo_cfg["actor"]["activation_fn"]),
        }
        self.actor = MLPGaussianActor(**actor_kwarg)
        self.actor_optim = getattr(optim, self.algo_cfg["actor"]["optimizer"])(
            self.actor.parameters(), self.algo_cfg["actor"]["lr"]
        )

        # critic
        critic_kwarg = {
            "input_shape": (self.state_shape[0] + self.action_shape[0],),
            "net_arch": self.algo_cfg["critic"]["net_arch"],
            "output_shape": (1,),
            "activation_fn": getattr(nn, self.algo_cfg["critic"]["activation_fn"]),
        }
        self.critic = MLPTwinCritic(**critic_kwarg)
        self.critic_target = deepcopy(self.critic)
        self.critic_optim = getattr(optim, self.algo_cfg["critic"]["optimizer"])(
            self.critic.parameters(), self.algo_cfg["critic"]["lr"]
        )

        # alpha, we optimize log(alpha) because alpha should always be bigger than 0.
        if self.algo_cfg["log_alpha"]["auto_tune"]:
            self.log_alpha = th.tensor(
                [self.algo_cfg["log_alpha"]["init_value"]],
                device=self.device,
                requires_grad=True,
            )
            self.log_alpha_optim = getattr(
                optim, self.algo_cfg["log_alpha"]["optimizer"]
            )([self.log_alpha], self.algo_cfg["log_alpha"]["lr"])
            self.models.update(
                {"log_alpha": self.log_alpha, "log_alpha_optim": self.log_alpha_optim}
            )
        else:
            self.log_alpha = th.tensor(
                [self.algo_cfg["log_alpha"]["init_value"]], device=self.device
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
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg
    ) -> Union[Tuple[th.Tensor, th.Tensor], th.Tensor, np.ndarray]:
        """
        :param deterministic: whether sample from the action distribution or just the action mean.
        :param return_dtype_tensor: whether the returned data's dtype keeps to be torch.Tensor or numpy.ndarray
        :param return_log_prob: whether return log_prob
        """
        # Due to the squash operation, we need that keep_dtype_tensor == True here.
        if return_log_prob:
            action, log_prob = self.actor.sample(
                state, deterministic, True, True, self.device
            )
            # squash action
            log_prob -= th.sum(
                2 * (np.log(2.0) - action - F.softplus(-2 * action)),
                axis=-1,
                keepdims=True,
            )
        else:
            action = self.actor.sample(state, deterministic, True, False, self.device)
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

        if not keep_dtype_tensor:
            action, log_prob = tensor2ndarray((action, log_prob))

        return (action, log_prob) if return_log_prob else action

    def update(self) -> Dict:
        self.log_info = dict()
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
                if self.algo_cfg["log_alpha"]["auto_tune"]:
                    self._update_alpha(self._update_actor(states))
                else:
                    self._update_actor(states)
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.algo_cfg["critic"]["tau"],
                )

        return self.log_info

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
                keep_dtype_tensor=True,
                return_log_prob=True,
            )
            target_Q1, target_Q2 = self.critic_target(
                True, next_states, pred_next_actions
            )
            target_Q = th.min(target_Q1, target_Q2) - self.alpha * pred_next_log_pis
            target_Q = rewards + self.gamma * (1 - dones) * target_Q

        Q1, Q2 = self.critic(True, states, actions)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.log_info.update(
            {"loss/critic": gradient_descent(self.critic_optim, critic_loss)}
        )

    def _update_actor(self, states: th.Tensor):
        pred_actions, pred_log_pis = self.select_action(
            states, deterministic=False, keep_dtype_tensor=True, return_log_prob=True
        )
        Q1, Q2 = self.critic(True, states, pred_actions)
        actor_loss = th.mean(self.alpha * pred_log_pis - th.min(Q1, Q2))
        self.log_info.update(
            {"loss/actor": gradient_descent(self.actor_optim, actor_loss)}
        )

        return pred_log_pis.detach()

    def _update_alpha(self, pred_log_pis: th.Tensor):
        """Auto-tune alpha
        
        Note: pred_log_pis are detached from the computation graph
        """
        alpha_loss = th.mean(self.log_alpha * (-pred_log_pis - self.entropy_target))
        self.log_info.update(
            {"loss/alpha": gradient_descent(self.log_alpha_optim, alpha_loss)}
        )

        # update alpha
        self.models["log_alpha"].data = self.log_alpha.data
