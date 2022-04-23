import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions.normal import Normal
from algo.rl.trpo import TRPOAgent


class PPOAgent(TRPOAgent):
    """Proximal Policy Optimization"""

    def __init__(self, configs: dict):
        super().__init__(configs)
        self.value_coef = configs.get("value_coef")
        self.entropy_coef = configs.get("entropy_coef")
        self.update_times = configs.get("update_times")
        self.batch_size = configs.get("batch_size")
        self.max_grad_norm = configs.get("max_grad_norm")
        self.epsilon_clip = configs.get("epsilon_clip")

        self.optim = Adam(
            [
                {"params": self.actor.parameters(), "lr": configs.get("actor_lr")},
                {"params": self.critic.parameters(), "lr": configs.get("critic_lr")},
            ]
        )
        self.models["optim"] = self.optim

    def update_param(self, states, actions, rewards, next_states, not_dones):
        # estimate advantage
        with torch.no_grad():
            Rs, advantages = self.gae(
                self.critic, states, rewards, next_states, not_dones
            )
            action_mean, action_std = self.actor(states)
            old_action_distribution = Normal(action_mean, action_std)
            old_log_action_probs = torch.sum(
                old_action_distribution.log_prob(actions), axis=-1, keepdims=True
            )
        # update actor and critic
        all_loss = np.array([])
        full_idx = np.array(list(range(self.replay_buffer.size)))
        for _ in range(self.update_times):
            np.random.shuffle(full_idx)
            for idx in np.split(full_idx, self.replay_buffer.size // self.batch_size):
                # critic loss
                values = self.critic(states[idx])
                critic_loss = F.mse_loss(values, Rs[idx])
                # actor loss
                action_mean, action_std = self.actor(states[idx])
                action_distribution = Normal(action_mean, action_std)
                log_action_probs = torch.sum(
                    action_distribution.log_prob(actions[idx]), axis=-1, keepdims=True
                )

                ratio = torch.exp(log_action_probs - old_log_action_probs[idx])
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
                all_loss = np.append(all_loss, loss.item())

        return {
            "mean_loss": np.mean(all_loss),
        }
