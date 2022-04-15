import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from algo.base import BaseAgent
from net.discriminator import GAILDiscrim, AIRLDiscrim
from algo.rl.ppo import PPOAgent
from utils.config import load_yml_config
from utils.buffer import ImitationReplayBuffer, SimpleReplayBuffer


class GAILAgent(BaseAgent):
    def __init__(self, configs, disc_type="gail"):
        super().__init__(configs)
        assert disc_type in ['gail', 'airl']
        
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

        if disc_type == 'gail':
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
        else:
            # discriminator
            self.disc = AIRLDiscrim(
                self.state_dim, configs["discriminator_hidden_size"], gamma=self.gamma, activation_fn=nn.ReLU
            ).to(
                self.device
            )  # output the probability of beign expert decision of (s, a)
            self.disc_optim = Adam(self.disc.parameters(), lr=configs["discriminator_lr"])
            self.expert_buffer = SimpleReplayBuffer(
                self.state_dim, self.action_dim, self.device, configs["expert_buffer_size"]
            )
        
        self.models = {
            "disc": self.disc,
            "disc_optim": self.disc_optim,
            **self.policy.models,
        }

    def _rollout(self):
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
            
    def _update_disc(self):
        all_disc_loss = np.array([])
        for _ in range(self.update_disc_times):
            # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
            pi_states, pi_actions = self.policy.replay_buffer.sample(self.batch_size)[
                :2
            ]
            d_pi = self.disc(pi_states, pi_actions)
            loss_pi = -F.logsigmoid(-d_pi).mean()

            exp_states, exp_actions = self.expert_buffer.sample(self.batch_size)[:2]
            d_exp = self.disc(exp_states, exp_actions)
            loss_exp = -F.logsigmoid(d_exp).mean()

            disc_loss = loss_exp + loss_pi
            self.disc_optim.zero_grad()
            disc_loss.backward()
            self.disc_optim.step()
            
            all_disc_loss = np.append(all_disc_loss, disc_loss.item())
            
        return {
            "disc_loss": np.mean(all_disc_loss)
        }
        
    def _update_gen(self):
        (
            states,
            actions,
            next_states,
            _,
            not_dones,
        ) = self.policy.replay_buffer.sample()
        rewards = self.disc.gail_reward(states, actions)
        return self.policy.update_param(
            states, actions, next_states, rewards, not_dones
        ) 
        
    def __call__(self, state, training=False):
        return self.policy(state, training=training)

    def update_param(self):
        # sample trajectories
        self._rollout()

        # update discriminator
        disc_snapshot = self._update_disc()

        # update generator
        policy_snapshot = self._update_gen()
        
        # clear replay buffer, for on-policy learning
        self.policy.replay_buffer.clear()

        return {**disc_snapshot, **policy_snapshot}

    def learn(self):
        return self.update_param()
