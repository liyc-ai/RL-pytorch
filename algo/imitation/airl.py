import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from algo.imitation.gail import GAILAgent
from net.discriminator import AIRLDiscrim
from utils.buffer import SimpleReplayBuffer


class AIRLAgent(GAILAgent):
    def __init__(self, configs: dict):
        super().__init__(configs)
        self.disc = AIRLDiscrim(
            self.state_dim,
            configs["discriminator_hidden_size"],
            gamma=self.gamma,
            activation_fn=nn.ReLU,
        ).to(
            self.device
        )  # output the probability of beign expert decision of (s, a)
        self.disc_optim = Adam(self.disc.parameters(), lr=configs["discriminator_lr"])
        self.expert_buffer = SimpleReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.device,
            configs["expert_buffer_size"],
        )
        self.models = {
            **self.models,
            **{
                "disc": self.disc,
                "disc_optim": self.disc_optim,
            },
        }

    def _get_log_pis(self, states, actions):
        mu, std = self.policy.actor(states)
        pi_dist = Normal(mu, std)
        log_pis = torch.sum(pi_dist.log_prob(actions), axis=-1, keepdims=True)
        return log_pis

    def _update_disc(self):
        all_disc_loss = np.array([])
        for _ in range(self.update_disc_times):
            # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
            (
                pi_states,
                pi_actions,
                _,
                pi_next_states,
                pi_not_dones,
            ) = self.policy.replay_buffer.sample(self.batch_size)
            with torch.no_grad():
                pi_log_pis = self._get_log_pis(pi_states, pi_actions)
            d_pi = self.disc(pi_states, pi_next_states, pi_not_dones, pi_log_pis)
            loss_pi = -F.logsigmoid(-d_pi).mean()

            (
                exp_states,
                _,
                exp_log_pis,
                exp_next_states,
                exp_not_dones,
            ) = self.expert_buffer.sample(self.batch_size)
            d_exp = self.disc(exp_states, exp_next_states, exp_not_dones, exp_log_pis)
            loss_exp = -F.logsigmoid(d_exp).mean()

            disc_loss = torch.mean(loss_exp + loss_pi)
            self.disc_optim.zero_grad()
            disc_loss.backward()
            self.disc_optim.step()

            all_disc_loss = np.append(all_disc_loss, disc_loss.item())

        return {"disc_loss": np.mean(all_disc_loss)}

    def _update_gen(self):
        (
            states,
            actions,
            _,
            next_states,
            not_dones,
        ) = self.policy.replay_buffer.sample()
        with torch.no_grad():
            log_pis = self._get_log_pis(states, actions)
        rewards = self.disc.airl_reward(states, next_states, not_dones, log_pis)
        return self.policy.update_param(
            states, actions, rewards, next_states, not_dones
        )
