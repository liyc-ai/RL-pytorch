from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.critic import Critic


class GAILDiscrim(Critic):
    def __init__(self, state_dim, hidden_size, action_dim, activation_fn=nn.Tanh):
        super().__init__(
            state_dim, hidden_size, action_dim, activation_fn, output_dim=1
        )

    def gail_reward(self, state, action):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D(s,a))].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(state, action))


class AIRLDiscrim(nn.Module):
    def __init__(self, state_dim, hidden_size, gamma, activation_fn=nn.Tanh):
        super().__init__()
        self.g = Critic(state_dim, hidden_size, activation_fn=activation_fn)
        self.h = Critic(state_dim, hidden_size, activation_fn=activation_fn)
        self.gamma = gamma

    def f(self, states, next_states, not_dones):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * not_dones * next_vs - vs

    def forward(self, states, next_states, not_dones, log_pis):
        return self.f(states, next_states, not_dones) - log_pis

    def airl_reward(self, states, next_states, not_dones, log_pis):
        # Discriminator's output is sigmoid(f - log_pi).
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, next_states, not_dones, log_pis))
