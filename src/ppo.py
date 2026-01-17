import random

import torch as th
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import BatchSampler

from src.utils.drls.gae import GAE
from src.utils.net.actor import MLPGaussianActor
from src.utils.net.critic import MLPCritic
from src.utils.net.ptu import gradient_descent, move_device

from .trpo import TRPOAgent


class PPOAgent(TRPOAgent):
    """Proximal Policy Optimization (PPO)"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def setup_model(self):
        # hyper-param
        self.epsilon = self.cfg.agent.epsilon
        self.lambda_ = self.cfg.agent.lambda_

        # GAE
        self.gae = GAE(
            self.gamma,
            self.lambda_,
            self.cfg.agent.norm_adv,
            self.cfg.agent.use_td_lambda,
        )

        # actor
        actor_kwarg = {
            "state_shape": self.state_shape,
            "net_arch": self.cfg.agent.actor.net_arch,
            "action_shape": self.action_shape,
            "activation_fn": getattr(nn, self.cfg.agent.actor.activation_fn),
        }
        self.actor = th.compile(MLPGaussianActor(**actor_kwarg))
        self.actor_optim = getattr(optim, self.cfg.agent.actor.optimizer)(
            self.actor.parameters(), self.cfg.agent.actor.lr
        )

        # value network
        value_net_kwarg = {
            "input_shape": self.state_shape,
            "output_shape": (1,),
            "net_arch": self.cfg.agent.value_net.net_arch,
            "activation_fn": getattr(nn, self.cfg.agent.value_net.activation_fn),
        }
        self.value_net = th.compile(MLPCritic(**value_net_kwarg))
        self.value_net_optim = getattr(optim, self.cfg.agent.value_net.optimizer)(
            self.value_net.parameters(), self.cfg.agent.value_net.lr
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
        for _ in range(self.cfg.agent.value_net.n_update):
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
                    - self.cfg.agent.entropy_coef * sampled_action_dist.entropy().mean()
                )
                self.stats.update(
                    {
                        "loss/actor": gradient_descent(
                            self.actor_optim,
                            loss,
                            self.actor.parameters(),
                            # experimental results show that clipping grad realy improves performance
                            self.cfg.agent.actor.clip,
                        )
                    }
                )
