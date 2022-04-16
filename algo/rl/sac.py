import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from algo.base import BaseAgent
from net.critic import Critic
from net.actor import StochasticActor
from torch.distributions.normal import Normal
from utils.net import soft_update
from utils.buffer import SimpleReplayBuffer


class SACAgent(BaseAgent):
    """Soft Actor Critic"""

    def __init__(self, configs):
        super().__init__(configs)
        self.gamma = configs["gamma"]
        self.env_steps = configs["env_steps"]
        self.start_timesteps = configs["start_timesteps"]
        self.rho = configs["rho"]
        self.fixed_alpha = configs["fixed_alpha"]
        self.entropy_target = -self.action_dim

        self.log_alpha = torch.zeros(
            1, device=self.device, requires_grad=True
        )  # We optimize log(alpha) because alpha should always be bigger than 0.
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=configs["alpha_lr"])
        with torch.no_grad():
            if self.fixed_alpha:
                self.alpha = configs["alpha"]  # entropy coefficient
            else:
                self.alpha = self.log_alpha.exp().item()

        # actor
        self.actor = StochasticActor(
            self.state_dim, configs["actor_hidden_size"], self.action_dim, nn.ReLU
        ).to(self.device)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=configs["actor_lr"]
        )
        # Q1
        self.critic_1 = Critic(
            self.state_dim, configs["critic_hidden_size"], self.action_dim, nn.ReLU
        ).to(self.device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        # Q2
        self.critic_2 = Critic(
            self.state_dim, configs["critic_hidden_size"], self.action_dim, nn.ReLU
        ).to(self.device)
        self.critic_target_2 = copy.deepcopy(self.critic_2)

        self.critic_params = itertools.chain(
            self.critic_1.parameters(), self.critic_2.parameters()
        )
        self.critic_optim = torch.optim.Adam(
            self.critic_params, lr=configs["critic_lr"]
        )

        self.replay_buffer = SimpleReplayBuffer(
            self.state_dim, self.action_dim, self.device, self.configs["buffer_size"]
        )

        self.models = {
            "actor": self.actor,
            "actor_optim": self.actor_optim,
            "critic_1": self.critic_1,
            "critic_target_1": self.critic_target_1,
            "critic_2": self.critic_2,
            "critic_target_2": self.critic_target_2,
            "critic_optim": self.critic_optim,
            "log_alpha": self.log_alpha,
            "alpha_optim": self.alpha_optim,
        }

        self.all_actor_loss, self.all_critic_loss, self.all_alpha_loss = (
            np.array([]),
            np.array([]),
            np.array([]),
        )

    def __call__(self, state, training=False, calcu_log_prob=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device) if type(state) == np.ndarray else state
        mu, std = self.actor(state)
        pi_dist = Normal(mu, std)
        if training:
            action = pi_dist.rsample()
        else:
            action = mu
        log_prob = 0.
        if calcu_log_prob:
            # calculate log pi, which is equivalent to Eq 26 in SAC paper, but more numerically stable
            log_prob = pi_dist.log_prob(action).sum(axis=-1, keepdims=True)
            log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(
                axis=1, keepdims=True
            )
        action = self.squash_action(action)
        return action, log_prob
        

    def update_param(self, states, actions, rewards, next_states, not_dones):
        # calculate target q value
        with torch.no_grad():
            next_actions, next_log_pis = self(
                next_states, training=True, calcu_log_prob=True
            )
            target_Q1, target_Q2 = self.critic_target_1(
                next_states, next_actions
            ), self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
            target_Q = rewards + not_dones * self.gamma * target_Q

        # update critic
        current_Q1, current_Q2 = self.critic_1(states, actions), self.critic_2(
            states, actions
        )
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update actor
        self.critic_1.eval(), self.critic_2.eval()  # Freeze Q-networks to save computational effort

        pred_actions, pred_log_pis = self(states, training=True, calcu_log_prob=True)
        current_Q1, current_Q2 = self.critic_1(states, pred_actions), self.critic_2(
            states, pred_actions
        )
        actor_loss = (
            self.alpha * pred_log_pis - torch.min(current_Q1, current_Q2)
        ).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_1.train(), self.critic_2.train()

        # update alpha
        if not self.fixed_alpha:
            pred_log_pis += self.entropy_target
            alpha_loss = -(self.log_alpha * pred_log_pis.detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.models[
                "log_alpha"
            ].data = self.log_alpha.data  # update log_alpha in self.models
            self.alpha = self.log_alpha.clone().detach().exp().item()

        # update target critic
        soft_update(self.rho, self.critic_1, self.critic_target_1)
        soft_update(self.rho, self.critic_2, self.critic_target_2)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item() if not self.fixed_alpha else 0.0,
        }

    def learn(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if (
            self.replay_buffer.size < self.start_timesteps
            or (self.replay_buffer.size - self.start_timesteps) % self.env_steps != 0
        ):
            return None  # update model after every env steps

        all_actor_loss, all_critic_loss, all_alpha_loss = (
            np.array([]),
            np.array([]),
            np.array([]),
        )

        for _ in range(self.configs["env_steps"]):
            (
                states,
                actions,
                rewards,
                next_states,
                not_dones,
            ) = self.replay_buffer.sample(self.configs["batch_size"])

            snapshot = self.update_param(
                states, actions, rewards, next_states, not_dones
            )
            all_actor_loss = np.append(all_actor_loss, snapshot["actor_loss"])
            all_critic_loss = np.append(all_critic_loss, snapshot["critic_loss"])
            all_alpha_loss = np.append(all_alpha_loss, snapshot["alpha_loss"])

        return {
            "mean_actor_loss": np.mean(all_actor_loss),
            "mean_critic_loss": np.mean(all_critic_loss),
            "mean_alpha_loss": np.mean(all_alpha_loss),
        }

    def load_model(self, model_path):
        """Overload the super load_model, due to log_alpha"""
        super().load_model(model_path)
        self.log_alpha.data = self.models["log_alpha"].data
        self.alpha = self.log_alpha.clone().detach().exp().item()
