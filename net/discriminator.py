import torch
import torch.nn as nn
import torch.nn.functional as F
from net.critic import Critic


class GAILDiscrim(Critic):
    def __init__(
        self, state_dim, hidden_size, action_dim, activation_fn=nn.Tanh, init=False
    ):
        super().__init__(
            state_dim, hidden_size, action_dim, activation_fn, output_dim=1, init=init
        )

    def gail_reward(self, state, action):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D(s,a))].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(state, action))
