"""Deep Deterministic Policy Gradient
- Lillicrap T P, Hunt J J, Pritzel A, et al. Continuous control with deep reinforcement learning[C]
//ICLR (Poster). 2016.
- Code modified from: https://github.com/sfujim/TD3
"""
import torch
import torch.nn.functional as F
import copy
import numpy as np
from algo.base import BaseAgent
from net.actor import DeterministicActor
from net.critic import Critic
from utils.net import soft_update
from utils.buffer import SimpleReplayBuffer


class DDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient"""

    def __init__(self, configs):
        super().__init__(configs)
        self.gamma = configs["gamma"]
        self.rho = configs["rho"]
        self.expl_std = configs["expl_std"] * self.action_high
        # actor
        self.actor = DeterministicActor(
            self.state_dim, configs["actor_hidden_size"], self.action_dim
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=configs["actor_lr"]
        )
        # critic
        self.critic = Critic(
            self.state_dim, configs["critic_hidden_size"], self.action_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=configs["critic_lr"]
        )
        self.replay_buffer = SimpleReplayBuffer(
            self.state_dim, self.action_dim, self.device, self.configs["buffer_size"]
        )
        self.models = {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "actor_optim": self.actor_optim,
            "critic": self.critic,
            "critic_target": self.critic_target,
            "critic_optim": self.critic_optim,
        }

    def __call__(self, state, training=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.squash_action(self.actor(state)).cpu().data.numpy().flatten()
            if training:
                action = (
                    action + np.random.normal(0, self.expl_std, size=self.action_dim)
                ).clip(-self.action_high, self.action_high)
        return action

    def update_param(self, states, actions, next_states, rewards, not_dones):
        # compute the target Q value
        with torch.no_grad():
            target_Q = self.critic_target(
                next_states, self.squash_action(self.actor_target(next_states))
            )
            target_Q = rewards + not_dones * self.gamma * target_Q

        # get current Q estimate
        current_Q = self.critic(states, actions)

        # compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # compute actor loss
        self.critic.eval()
        actor_loss = -torch.mean(
            self.critic(states, self.squash_action(self.actor(states)))
        )

        # optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic.train()

        # update the frozen target models
        soft_update(self.rho, self.critic, self.critic_target)
        soft_update(self.rho, self.actor, self.actor_target)

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def learn(self, state, action, reward, next_state, done):
        # update buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        if self.replay_buffer.size < self.configs["start_timesteps"]:
            return None  # warm buffer

        states, actions, rewards, next_states, not_dones = self.replay_buffer.sample(
            self.configs["batch_size"]
        )

        return self.update_param(states, actions, next_states, rewards, not_dones)
