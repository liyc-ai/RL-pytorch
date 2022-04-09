import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
from algo.base import BaseAgent
from network.actor import StochasticActor
from utils.buffer import ImitationReplayBuffer


class BCAgent(BaseAgent):
    """Behavioral Cloning"""

    def __init__(self, configs):
        super().__init__(configs)
        self.num_epochs = configs["num_epochs"]
        self.batch_size = configs["batch_size"]

        self.actor = StochasticActor(
            self.state_dim, configs["actor_hidden_size"], self.action_dim
        ).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=configs["actor_lr"])

        self.replay_buffer = ImitationReplayBuffer(
            self.state_dim, self.action_dim, self.device, configs["buffer_size"]
        )
        
        self.models = {
            "actor": self.actor,
            "optim": self.actor_optim,
        }

    def select_action(self, state, training=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action_mean, action_std = self.actor(state)
            if training:
                action = torch.normal(action_mean, action_std)
            else:
                action = action_mean
        return action.cpu().data.numpy().flatten()

    def learn(self, trajectory):
        # insert new trajectory into replay buffer
        observations, actions = trajectory["observations"], trajectory["actions"]
        traj_len = actions.shape[0]
        for i in range(traj_len):
            self.replay_buffer.add(observations[i], actions[i])

        # train actor
        for _ in range(self.num_epochs):
            self.batch_size = min(self.replay_buffer.size, self.batch_size)
            states, actions = self.replay_buffer.sample(self.batch_size)
            action_mean, action_std = self.actor(states)
            log_prob = Normal(action_mean, action_std).log_prob(actions)
            loss = -log_prob.mean()
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
