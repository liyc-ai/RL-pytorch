import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.normal import Normal
from algo.base import BaseAgent
from network.actor import StochasticActor
from network.critic import Critic
from utils.gae import GAE


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization"""

    def __init__(self, configs):
        super().__init__(configs)

        self.rollout_steps = configs["rollout_steps"]
        self.lambda_ = configs["lambda"]
        self.epsilon_clip = configs["epsilon_clip"]
        self.value_coef = configs["value_coef"]
        self.entropy_coef = configs["entropy_coef"]
        self.update_times = configs["update_times"]
        self.mini_batch_size = configs["mini_batch_size"]
        self.max_grad_norm = configs["max_grad_norm"]

        self.gae = GAE(self.gamma, self.lambda_)
        self.actor = StochasticActor(
            self.state_dim,
            configs["actor_hidden_size"],
            self.action_dim,
            state_std_independent=True,
            init=True,
        ).to(self.device)
        self.critic = Critic(
            self.state_dim, configs["critic_hidden_size"], init=True
        ).to(self.device)
        self.optim = Adam(
            [
                {"params": self.actor.parameters(), "lr": configs["actor_lr"]},
                {"params": self.critic.parameters(), "lr": configs["critic_lr"]},
            ]
        )

        self.models = {
            "actor": self.actor,
            "critic": self.critic,
            "optim": self.optim,
        }

    def select_action(self, state, training=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action_mean, action_std = self.actor(state)
            if training:
                action = torch.normal(action_mean, action_std)
            else:
                action = action_mean
        return action.cpu().data.numpy().flatten()

    def learn(self, state, action, next_state, reward, done):
        # collect transitions
        self.replay_buffer.add(state, action, next_state, reward, done)
        if self.replay_buffer.size < self.rollout_steps:
            return

        # estimate advantage
        states, actions, next_states, rewards, not_dones = self.replay_buffer.sample()
        with torch.no_grad():
            Rs, advantages = self.gae(
                self.critic, states, rewards, not_dones, next_states
            )

            action_mean, action_std = self.actor(states)
            fixed_action_distribution = Normal(action_mean, action_std)
            fixed_log_action_probs = fixed_action_distribution.log_prob(actions).sum(
                axis=-1, keepdims=True
            )

        # update actor and critic
        full_idx = np.array(list(range(self.replay_buffer.size)))
        for _ in range(self.update_times):
            np.random.shuffle(full_idx)
            for idx in np.split(
                full_idx, self.replay_buffer.size // self.mini_batch_size
            ):
                # critic loss
                values = self.critic(states[idx])
                critic_loss = F.mse_loss(Rs[idx], values)
                # actor loss
                action_mean, action_std = self.actor(states[idx])
                action_distribution = Normal(action_mean, action_std)
                log_action_probs = action_distribution.log_prob(actions[idx]).sum(
                    axis=-1, keepdims=True
                )

                ratio = torch.exp(log_action_probs - fixed_log_action_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip)
                    * advantages[idx]
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                # entropy bonus
                entropy = -action_distribution.entropy().mean()
                # total loss
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy
                )
                self.optim.zero_grad()
                loss.backward()
                # gradient clip
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optim.step()

        # clear buffer
        self.replay_buffer.clear()
