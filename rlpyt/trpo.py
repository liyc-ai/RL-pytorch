import random
from typing import Callable, Dict, Tuple, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from drlplugs.drls.gae import GAE
from drlplugs.net.actor import MLPGaussianActor
from drlplugs.net.critic import MLPCritic
from drlplugs.net.ptu import gradient_descent, move_device
from omegaconf import DictConfig
from torch import nn, optim
from torch.autograd import grad
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.utils.data import BatchSampler

from rlpyt import BaseRLAgent


class TRPOAgent(BaseRLAgent):
    """Trust Region Policy Optimization (TRPO)"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def setup_model(self):
        # hyper-param
        self.lambda_ = self.cfg.agent.lambda_

        ## conjugate gradient
        self.residual_tol = self.cfg.agent.residual_tol
        self.cg_steps = self.cfg.agent.cg_steps
        self.damping = self.cfg.agent.damping

        ## line search
        self.beta = self.cfg.agent.beta
        self.max_backtrack = self.cfg.agent.max_backtrack
        self.accept_ratio = self.cfg.agent.accept_ratio
        self.delta = self.cfg.agent.delta

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
        self.actor = MLPGaussianActor(**actor_kwarg)

        # value network
        value_net_kwarg = {
            "input_shape": self.state_shape,
            "output_shape": (1,),
            "net_arch": self.cfg.agent.value_net.net_arch,
            "activation_fn": getattr(nn, self.cfg.agent.value_net.activation_fn),
        }
        self.value_net = MLPCritic(**value_net_kwarg)
        self.value_net_optim = getattr(optim, self.cfg.agent.value_net.optimizer)(
            self.value_net.parameters(), self.cfg.agent.value_net.lr
        )

        move_device((self.actor, self.value_net), self.device)

        self.models.update(
            {
                "actor": self.actor,
                "value_net": self.value_net,
                "value_net_optim": self.value_net_optim,
            }
        )

    def select_action(
        self, state: np.ndarray, deterministic: bool, return_log_prob: bool, **kwarg
    ) -> Union[Tuple[th.Tensor, th.Tensor], th.Tensor]:
        return self.actor.sample(state, deterministic, return_log_prob, self.device)

    def update(self) -> Dict:
        self.log_info = dict()
        if self.trans_buffer.size >= self.cfg.agent.rollout_steps:
            states, actions, next_states, rewards, dones = self.trans_buffer.buffers

            # get advantage
            with th.no_grad():
                Rs, self.adv = self.gae(
                    self.value_net, states, rewards, next_states, dones
                )

            self._update_actor(states, actions)
            self._update_value_net(states, Rs)

            self.trans_buffer.clear()
        return self.log_info

    def _update_value_net(self, states: th.Tensor, Rs: th.Tensor) -> float:
        idx = list(range(self.trans_buffer.size))
        for _ in range(self.cfg.agent.value_net.n_update):
            random.shuffle(idx)
            batches = list(
                BatchSampler(idx, batch_size=self.batch_size, drop_last=False)
            )
            for batch in batches:
                sampled_states = states[batch]
                values = self.value_net(sampled_states)
                loss = F.mse_loss(values, Rs[batch])
                self.log_info.update(
                    {"loss/critic": gradient_descent(self.value_net_optim, loss)}
                )

    def _update_actor(self, states: th.Tensor, actions: th.Tensor):
        original_actor_param = th.clone(
            parameters_to_vector(self.actor.parameters()).data
        )

        ## pg
        action_dist, log_probs = self._select_action_dist(states, actions)
        old_action_dist = Normal(
            action_dist.loc.data.clone(), action_dist.scale.data.clone()
        )
        old_log_probs = log_probs.data.clone()

        loss = self._get_surrogate_loss(log_probs, old_log_probs)
        pg = grad(loss, self.actor.parameters(), retain_graph=True)
        pg = parameters_to_vector(pg).detach()

        ## x = H^{-1} * pg, H = kl_g'
        kl = th.mean(kl_divergence(old_action_dist, action_dist))
        kl_g = grad(kl, self.actor.parameters(), create_graph=True)
        kl_g = parameters_to_vector(kl_g)

        update_dir = self._conjugate_gradient(kl_g, pg)  # x
        Fvp = self._Fvp_func(kl_g, pg)  # Hx
        full_step_size = th.sqrt(
            2 * self.delta / th.dot(update_dir, Fvp)
        )  # denominator: x^t (Hx)

        ## line search for appropriate step size
        self.log_info.update({"loss/actor": 0.0})

        def check_constrain(alpha):
            step = alpha * full_step_size * update_dir
            with th.no_grad():
                vector_to_parameters(
                    original_actor_param + step, self.actor.parameters()
                )
                try:
                    new_action_dist, new_log_probs = self._select_action_dist(
                        states, actions
                    )
                except:
                    vector_to_parameters(  # restore actor
                        original_actor_param, self.actor.parameters()
                    )
                    return False
                new_loss = self._get_surrogate_loss(new_log_probs, old_log_probs)
                new_kl = th.mean(kl_divergence(old_action_dist, new_action_dist))
                actual_improve = new_loss - loss

            if actual_improve.item() > 0.0 and new_kl.item() <= self.delta:
                self.log_info.update({"loss/actor": new_loss.item()})
                return True
            else:
                return False

        alpha = self._line_search(check_constrain)
        vector_to_parameters(
            original_actor_param + alpha * full_step_size * update_dir,
            self.actor.parameters(),
        )

    def _select_action_dist(
        self, states: th.Tensor, actions: th.Tensor
    ) -> Tuple[Normal, th.Tensor]:
        action_mean, action_std = self.actor(states)
        action_dist = Normal(action_mean, action_std)
        log_prob = th.sum(action_dist.log_prob(actions), dim=-1, keepdim=True)
        return action_dist, log_prob

    def _line_search(self, check_constrain: Callable) -> float:
        alpha = 1.0 / self.beta
        for _ in range(self.max_backtrack):
            alpha *= self.beta
            if check_constrain(alpha):
                return alpha
        return 0.0

    def _get_surrogate_loss(
        self, log_probs: th.Tensor, old_log_probs: th.Tensor
    ) -> th.Tensor:
        return th.mean(th.exp(log_probs - old_log_probs) * self.adv)

    def _conjugate_gradient(self, kl_g: th.Tensor, pg: th.Tensor) -> th.Tensor:
        """To calculate s = H^{-1}g without solving inverse of H

        Ref: https://en.wikipedia.org/wiki/Conjugate_gradient_method

        Code modified from: https://github.com/ikostrikov/pytorch-trpo
        """
        x = th.zeros_like(pg)
        r = pg.clone()
        p = pg.clone()
        rdotr = th.dot(r, r)
        for _ in range(self.cg_steps):
            _Fvp = self._Fvp_func(kl_g, p)
            alpha = rdotr / th.dot(p, _Fvp)
            x += alpha * p
            r -= alpha * _Fvp
            new_rdotr = th.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < self.residual_tol:
                break
        return x

    def _Fvp_func(self, kl_g: th.Tensor, p: th.Tensor) -> th.Tensor:
        """Fisher vector product"""
        gvp = th.dot(kl_g, p)
        Hvp = grad(gvp, self.actor.parameters(), retain_graph=True)
        Hvp = parameters_to_vector(Hvp).detach()
        # tricks to stablize
        # see https://www2.maths.lth.se/matematiklth/vision/publdb/reports/pdf/byrod-eccv-10.pdf
        Hvp += self.damping * p
        return Hvp
