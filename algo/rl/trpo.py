import numpy as np
import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from algo.base import BaseAgent
from net.actor import StochasticActor
from net.critic import Critic
from utils.gae import GAE
from utils.buffer import SimpleReplayBuffer


class TRPOAgent(BaseAgent):
    """Trust Region Policy Optimization"""

    def __init__(self, configs: dict):
        super().__init__(configs)
        self.gamma = configs.get("gamma")
        self.rollout_steps = configs.get("rollout_steps")
        self.lambda_ = configs.get("lambda")
        self.residual_tol = configs.get("residual_tol")
        self.cg_steps = configs.get("cg_steps")
        self.damping = configs.get("damping")
        self.delta = configs.get("delta")
        self.beta = configs.get("beta")
        self.max_backtrack = configs.get("max_backtrack")
        self.line_search_accept_ratio = configs.get("line_search_accept_ratio")
        self.n_critic_update = configs.get("n_critic_update")

        self.replay_buffer = SimpleReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.device,
            configs.get("buffer_size"),
        )

        self.gae = GAE(self.gamma, self.lambda_)

        self.actor = StochasticActor(
            self.state_dim, configs.get("actor_hidden_size"), self.action_dim
        ).to(self.device)
        self.critic = Critic(self.state_dim, configs.get("critic_hidden_size")).to(
            self.device
        )
        self.optim = Adam(
            self.critic.parameters(),
            configs.get("critic_lr"),
            weight_decay=configs.get("weight_decay"),
        )
        self.models = {
            "actor": self.actor,
            "critic": self.critic,
            "optim": self.optim,
        }

    def _conjugate_gradient(self, Hvp_func, g):
        """To calculate s = H^{-1}g without solving inverse of H

        Code modified from: https://github.com/ikostrikov/pytorch-trpo
        """
        x = torch.zeros_like(g)
        r = g.clone()
        p = g.clone()
        rdotr = torch.dot(r, r)
        for _ in range(self.cg_steps):
            _Hvp = Hvp_func(p)
            alpha = rdotr / torch.dot(p, _Hvp)
            x += alpha * p
            r -= alpha * _Hvp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < self.residual_tol:
                break
        return x

    def _calcu_surrogate_loss(self, log_action_probs):
        return torch.mean(
            torch.exp(log_action_probs - self.old_log_action_probs) * self.advantages
        )

    def _calcu_sample_average_kl(self, mus, stds):
        action_distribution = Normal(mus, stds)
        return torch.mean(
            kl_divergence(self.old_action_distribution, action_distribution)
        )  # sample average kl-divergence

    def _line_search(self, update_dir, full_step_size, check_constraints_satisfied):
        """https://en.wikipedia.org/wiki/Backtracking_line_search"""
        alpha = full_step_size / self.beta

        for _ in range(self.max_backtrack):
            alpha *= self.beta
            if check_constraints_satisfied(alpha * update_dir, alpha):
                return alpha

        return 0.0

    def _apply_update(self, update):
        """Apply update to actor

        Code modified from: torch.nn.utils.convert_parameters.vector_to_parameters
        """
        n = 0
        for param in self.actor.parameters():
            numel = param.numel()
            param_update = update[n : n + numel].view(param.size())
            param.data += param_update
            n += numel

    def _Hvp_func(self, v):
        gvp = torch.sum(self.grads * v)
        Hvp = (
            parameters_to_vector(grad(gvp, self.actor.parameters(), retain_graph=True))
            .clone()
            .detach()
        )
        Hvp += self.damping * v
        return Hvp

    def __call__(self, state, training=False, calcu_log_prob=False):
        state = (
            torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if type(state) == np.ndarray
            else state
        )
        action_mean, action_std = self.actor(state)
        return self.select_action(action_mean, action_std, training, calcu_log_prob)

    def update_param(self, states, actions, rewards, next_states, not_dones):
        # estimate advantage
        with torch.no_grad():
            Rs, self.advantages = self.gae(
                self.critic, states, rewards, next_states, not_dones
            )

        # estimate actor gradient
        action_mean, action_std = self.actor(states)
        action_distribution = Normal(action_mean, action_std)
        log_action_probs = torch.sum(
            action_distribution.log_prob(actions), axis=-1, keepdims=True
        )

        self.old_action_distribution = Normal(
            action_mean.clone().detach(), action_std.clone().detach()
        )
        self.old_log_action_probs = torch.sum(
            self.old_action_distribution.log_prob(actions), axis=-1, keepdims=True
        )

        loss = self._calcu_surrogate_loss(log_action_probs)
        g = (
            parameters_to_vector(grad(loss, self.actor.parameters(), retain_graph=True))
            .clone()
            .detach()
        )  # flatten g into a single vector

        # Hessian vector product estimation
        kl = self._calcu_sample_average_kl(action_mean, action_std)
        self.grads = parameters_to_vector(
            grad(kl, self.actor.parameters(), create_graph=True)
        )

        update_dir = self._conjugate_gradient(self._Hvp_func, g)  # update direction
        Hvp = self._Hvp_func(update_dir)
        full_step_size = torch.sqrt(
            2 * self.delta / torch.dot(update_dir, Hvp)
        )  # expected update size

        # line search
        expected_improvement = torch.dot(g, update_dir)

        def check_constrained(update, alpha):
            with torch.no_grad():
                self._apply_update(update)
                new_action_mean, new_action_std = self.actor(states)
                try:
                    new_action_distribution = Normal(new_action_mean, new_action_std)
                except:
                    raise ValueError("Invalid Gradient!")
                new_log_action_probs = torch.sum(
                    new_action_distribution.log_prob(actions), axis=-1, keepdims=True
                )

                new_loss = self._calcu_surrogate_loss(new_log_action_probs)
                new_mean_kl = self._calcu_sample_average_kl(
                    new_action_mean, new_action_std
                )
                self._apply_update(-update)

            actual_improvement = new_loss - loss
            improvement_ratio = actual_improvement / (expected_improvement * alpha)
            surrogate_cond = (
                improvement_ratio >= self.line_search_accept_ratio
                and actual_improvement > 0.0
            )

            kl_cond = new_mean_kl <= self.delta

            return surrogate_cond and kl_cond

        real_step_size = self._line_search(
            update_dir, full_step_size, check_constrained
        )
        self._apply_update(real_step_size * update_dir)  # update actor

        # update critic
        all_critic_loss = np.array([])
        for _ in range(self.n_critic_update):
            values = self.critic(states)
            critic_loss = F.mse_loss(values, Rs)
            self.optim.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.optim.step()
            all_critic_loss = np.append(all_critic_loss, critic_loss.item())

        # clear buffer
        self.replay_buffer.clear()

        return {
            "surrogate_loss": loss.item(),
            "critic_loss": np.mean(all_critic_loss),
        }

    def learn(self, state, action, reward, next_state, done):
        # collect transitions
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.replay_buffer.size < self.rollout_steps:
            return None

        states, actions, rewards, next_states, not_dones = self.replay_buffer.sample()
        return self.update_param(states, actions, rewards, next_states, not_dones)
