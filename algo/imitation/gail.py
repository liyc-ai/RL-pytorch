import gym
import numpy as np
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
        self.policy = PPOAgent(expert_config)

        # discriminator
        self.disc = GAILDiscrim(
            self.state_dim, configs["critic_hidden_size"], self.action_dim
        ).to(self.device)
        self.disc_optim = Adam(self.disc.parameters(), lr=configs["discriminator_lr"])

        self.expert_buffer = ImitationReplayBuffer(
            self.state_dim, self.action_dim, self.device, configs["expert_buffer_size"]
        )

        self.models = {
            "policy": self.policy,
            "disc": self.disc,
        }

    def rollout(self):
        done = True
        for _ in range(self.policy.rollout_steps):
            if done:
                state = self.env.reset()
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)
            self.policy.replay_buffer.add(state, action, next_state, reward, done)
            state = next_state

    def __call__(self, state, training=False):
        return self.policy(state, training)

    def update_param(self):
        # sample trajectories
        self.rollout()
        # update discriminator
        all_disc_loss = np.array([])
        for _ in range(self.update_disc_times):
            # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
            pi_states, pi_actions = self.policy.replay_buffer.sample(self.batch_size)
            log_pi = self.disc(pi_states, pi_actions)
            loss_pi = -F.logsigmoid(-log_pi).mean()

            exp_states, exp_actions = self.expert_buffer.sample(self.batch_size)
            log_exp = self.disc(exp_states, exp_actions)
            loss_exp = -F.logsigmoid(log_exp).mean()

            disc_loss = loss_exp + loss_pi
            self.disc_optim.zero_grad()
            disc_loss.backward()
            self.disc_optim.step()

            np.append(all_disc_loss, disc_loss.item())

        # update generator
        (
            states,
            actions,
            next_states,
            _,
            not_dones,
        ) = self.policy.replay_buffer.sample()
        rewards = self.disc.gail_reward(states, actions)
        policy_loss = self.policy.update_param(
            states, actions, next_states, rewards, not_dones
        )

        return {
            "disc_loss": np.mean(all_disc_loss),
            "genetor_loss": policy_loss,
        }

    def learn(self):
        return self.update_param()
