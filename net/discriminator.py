import torch
import torch.nn as nn
import torch.nn.functional as F
from net.critic import Critic
from utils.transform import Normalizer


class GAILDiscrim(Critic):
    def __init__(self, state_dim, hidden_size, action_dim, activation_fn=nn.Tanh):
        super().__init__(
            state_dim, hidden_size, action_dim, activation_fn, output_dim=1
        )

        self.returns = None
        self.reward_normalizer = Normalizer()
        self.gamma = 0.99

    def gail_reward(self, state, action):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D(s,a))].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(state, action))
