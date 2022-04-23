import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.normal import Normal
from algo.base import BaseAgent
from net.actor import StochasticActor
from utils.buffer import ImitationReplayBuffer


class BCAgent(BaseAgent):
    """Behavior Cloning"""

    def __init__(self, configs: dict):
        super().__init__(configs)
        self.batch_size = configs.get("batch_size")
        self.max_grad_norm = configs.get("max_grad_norm")

        self.actor = StochasticActor(
            self.state_dim, configs.get("actor_hidden_size"), self.action_dim
        ).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=configs.get("actor_lr"))

        self.expert_buffer = ImitationReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.device,
            configs.get("expert_buffer_size"),
        )

        self.models = {
            "actor": self.actor,
            "optim": self.actor_optim,
        }

        self.mse_loss_fn = None
        if configs.get("loss_fn") == "mse":
            self.mse_loss_fn = nn.MSELoss()

    def __call__(self, state, training=False, calcu_log_prob=False):
        state = (
            torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if type(state) == np.ndarray
            else state
        )
        action_mean, action_std = self.actor(state)
        return self.select_action(action_mean, action_std, training, calcu_log_prob)

    def _bc_loss(self, buffer):
        states, actions = buffer.sample(self.batch_size)[:2]
        action_means, action_stds = self.actor(states)
        if self.mse_loss_fn != None:
            loss = self.mse_loss_fn(action_means, actions)
        else:
            log_probs = torch.sum(
                Normal(action_means, action_stds).log_prob(actions),
                axis=-1,
                keepdims=True,
            )
            loss = -torch.mean(log_probs)
        return loss

    def update_param(self, buffer):
        loss = self._bc_loss(buffer)
        self.actor_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        return {
            "loss": loss.item(),
        }

    def learn(self):
        return self.update_param(self.expert_buffer)
