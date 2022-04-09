import torch
from torch.optim import Adam
from algo.base import BaseAgent
from network.actor import StochasticActor


class BCAgent(BaseAgent):
    """Behavioral Cloning"""

    def __init__(self, configs):
        super().__init__(configs)
        self.actor = StochasticActor(
            self.state_dim, configs["actor_hidden_size"], self.action_dim
        )
        self.actor_optim = Adam(self.actor.parameters(), lr=configs["actor_lr"])

    def select_action(self, state, training=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action_mean, action_std = self.actor(state)
            if training:
                action = torch.normal(action_mean, action_std)
            else:
                action = action_mean
        return action.cpu().data.numpy().flatten()

    def learn(self, trajectory):
        ...
