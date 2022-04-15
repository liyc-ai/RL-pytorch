import torch
import numpy as np


class GAE:
    """Estimate Advantage using GAE (https://arxiv.org/abs/1506.02438)
    Ref:
    [1] https://nn.labml.ai/rl/ppo/gae.html
    [2] https://github.com/ikostrikov/pytorch-trpo
    """

    def __init__(self, gamma, lambda_):
        self.gamma = gamma
        self.lambda_ = lambda_

    def __call__(
        self, value_net, states, rewards, not_dones, next_states, use_td_lambd=True
    ):
        """Here we can use two different methods to calculate Returns"""
        if use_td_lambd:
            Rs, advantages = self.td_lambda(
                value_net, states, rewards, not_dones, next_states
            )
        else:
            Rs, advantages = self.gae(
                value_net, states, rewards, not_dones, next_states
            )

        return Rs, (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def gae(self, value_net, states, rewards, not_dones, next_states):
        Rs = torch.empty_like(rewards)  # reward-to-go R_t
        advantages = torch.empty_like(rewards)  # advantage
        values = value_net(states)

        last_value = value_net(next_states[-1])
        last_return = last_value.clone()
        last_advantage = 0.0

        for t in reversed(range(rewards.shape[0])):
            # calculate rewards-to-go reward
            Rs[t] = rewards[t] + self.gamma * last_return * not_dones[t]
            # delta and advantage
            delta = rewards[t] + self.gamma * last_value * not_dones[t] - values[t]
            advantages[t] = (
                delta + self.gamma * self.lambda_ * last_advantage * not_dones[t]
            )
            # update pointer
            last_value = values[t].clone()
            last_advantage = advantages[t].clone()
            last_return = Rs[t].clone()
        return Rs, advantages

    def td_lambda(self, value_net, states, rewards, not_dones, next_states):
        # Calcultae value
        values, next_values = value_net(states), value_net(next_states)
        # Calculate TD errors.
        deltas = rewards + self.gamma * next_values * not_dones - values
        # Initialize gae.
        advantages = torch.empty_like(rewards)
        # Calculate gae recursively from behind.
        advantages[-1] = deltas[-1]
        for t in reversed(range(rewards.size(0) - 1)):
            advantages[t] = deltas[t] + self.gamma * self.lambda_ * not_dones[t] * advantages[t + 1]

        return advantages + values, advantages
