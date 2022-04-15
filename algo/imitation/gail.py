import re
import gym
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from torch.optim import Adam
from algo.base import BaseAgent
from net.discriminator import GAILDiscrim
from algo.rl.ppo import PPOAgent
from utils.config import load_yml_config
from utils.buffer import ImitationReplayBuffer


class GAILAgent(BaseAgent):
    def __init__(self, configs):
        super().__init__(configs)
        self.update_disc_times = configs["update_disc_times"]
        self.batch_size = configs["batch_size"]
        self.env = gym.make(configs["env_name"])
        self.env.seed(configs["seed"])

        # use ppo to train generator
        expert_config = load_yml_config("ppo.yml")
        (
            expert_config["state_dim"],
            expert_config["action_dim"],
            expert_config["action_high"],
        ) = (self.state_dim, self.action_dim, self.action_high)
        if configs["expert"] is not None:
            expert_config = {
                **expert_config,
                **configs["expert"],
            }
        self.policy = PPOAgent(expert_config)
        self.gamma = self.policy.gamma

        # discriminator
        self.disc = GAILDiscrim(
            self.state_dim, configs["discriminator_hidden_size"], self.action_dim
        ).to(
            self.device
        )  # output the probability of beign expert decision of (s, a)
        self.disc_optim = Adam(self.disc.parameters(), lr=configs["discriminator_lr"])

        self.expert_buffer = ImitationReplayBuffer(
            self.state_dim, self.action_dim, self.device, configs["expert_buffer_size"]
        )

        self.models = {
            "disc": self.disc,
            "disc_optim": self.disc_optim,
            **self.policy.models,
        }

    def rollout(self):
        done = True
        for i in range(self.policy.rollout_steps):
            if done:
                next_state = self.env.reset()
            state = next_state
            action = self.policy(state, training=True)
            next_state, reward, done, _ = self.env.step(action)
            real_done = done if i < self.env._max_episode_steps else False
            self.policy.replay_buffer.add(
                state, action, next_state, reward, float(real_done)
            )

    def __call__(self, state, training=False):
        return self.policy(state, training=training)

    def update_param(self):
        # sample trajectories
        self.rollout()

        # update discriminator
        all_disc_loss = np.array([])
        for _ in range(self.update_disc_times):
            # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
            pi_states, pi_actions = self.policy.replay_buffer.sample(self.batch_size)[
                :2
            ]
            d_pi = self.disc(pi_states, pi_actions)
            loss_pi = -F.logsigmoid(-d_pi).mean()

            exp_states, exp_actions = self.expert_buffer.sample(self.batch_size)
            d_exp = self.disc(exp_states, exp_actions)
            loss_exp = -F.logsigmoid(d_exp).mean()

            disc_loss = loss_exp + loss_pi
            self.disc_optim.zero_grad()
            disc_loss.backward()
            self.disc_optim.step()

            all_disc_loss = np.append(all_disc_loss, disc_loss.item())

        # update generator
        (
            states,
            actions,
            next_states,
            _,
            not_dones,
        ) = self.policy.replay_buffer.sample()
        rewards = self.disc.gail_reward(states, actions)
        policy_snapshot = self.policy.update_param(
            states, actions, next_states, rewards, not_dones
        )

        self.policy.replay_buffer.clear()

        return {"disc_loss": np.mean(all_disc_loss), **policy_snapshot}

    def learn(self):
        return self.update_param()
