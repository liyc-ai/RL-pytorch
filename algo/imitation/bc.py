import torch
from torch.optim import Adam
from algo.imitation.base import BaseImitator
from network.actor import StochasticActor


class BCImitator(BaseImitator):
    """Behavioral Cloning
    """
    def __init__(self, configs):
        super().__init__(configs)
        self.actor = StochasticActor(self.state_dim, configs['actor_hidden_size'], self.action_dim)
        self.actor_optim = Adam(self.actor.parameters(), lr=configs['actor_lr'])
      
    def select_action(self, state, training=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        
    def learn(self, logger, writer):
        ...
        