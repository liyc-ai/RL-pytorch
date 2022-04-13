import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.normal import Normal
from algo.base import BaseAgent
from net.actor import StochasticActor
from utils.buffer import ImitationReplayBuffer


class BCAgent(BaseAgent):
    """Behavioral Cloning"""

    def __init__(self, configs):
        super().__init__(configs)
        self.batch_size = configs["batch_size"]
        self.max_grad_norm = configs["max_grad_norm"]

        self.actor = StochasticActor(
            self.state_dim, configs["actor_hidden_size"], self.action_dim
        ).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=configs["actor_lr"])

        self.expert_buffer = ImitationReplayBuffer(
            self.state_dim, self.action_dim, self.device, configs["expert_buffer_size"]
        )

        self.models = {
            "actor": self.actor,
            "optim": self.actor_optim,
        }

        # Note: we observe that mse loss works better than mle loss in BC
        self.mse_loss_fn = None
        if configs["loss_fn"] == "mse":
            self.mse_loss_fn = nn.MSELoss()

    def __call__(self, state, training=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action_mean, action_std = self.actor(state)
            if training:
                action = torch.normal(action_mean, action_std)
            else:
                action = action_mean
        return action.cpu().data.numpy().flatten()

    def _bc_loss(self):
        states, actions = self.expert_buffer.sample(self.batch_size)
        action_means, action_stds = self.actor(states)
        if self.mse_loss_fn != None:
            loss = self.mse_loss_fn(action_means, actions)
        else:
            log_prob = (
                Normal(action_means, action_stds)
                .log_prob(actions)
                .sum(axis=-1, keepdims=True)
            )
            loss = -log_prob.mean()
        return loss

    def update_param(self):
        loss = self._bc_loss()
        self.actor_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optim.step()

        return {
            "loss": loss.item(),
        }

    def learn(self):
        return self.update_param()
