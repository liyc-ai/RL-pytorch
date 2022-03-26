import torch
import torch.nn as nn 
from utils.net import build_mlp_extractor, weights_init_

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class DeterministicActor(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim, activation_fn=nn.ReLU, init=False):
        super().__init__()
        self.feature_extractor = nn.Sequential(*build_mlp_extractor(state_dim, hidden_size, activation_fn))
        
        if len(hidden_size)>0:
            input_dim = hidden_size[-1]
        else:
            input_dim = state_dim
            
        self.output_head = nn.Linear(input_dim, action_dim)
        
        if init:
            self.output_head.weight.data.mul_(0.1)
            self.output_head.bias.data.mul_(0.0)
        
        # self.apply(weights_init_)
        
    def forward(self, state):
        feature = self.feature_extractor(state)
        action = self.output_head(feature)
        return action

class StochasticActor(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim, activation_fn=nn.Tanh, state_std_independent=False, init=False):
        super().__init__()
        self.state_std_independent = state_std_independent
        self.feature_extractor = nn.Sequential(*build_mlp_extractor(state_dim, hidden_size, activation_fn))
        
        if len(hidden_size)>0:
            input_dim = hidden_size[-1]
        else:
            input_dim = state_dim 
        # mean and log std
        self.mu = nn.Linear(input_dim, action_dim)
        if state_std_independent:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        else:
            self.log_std = nn.Linear(input_dim, action_dim)
        
        if init:
            self.mu.weight.data.mul_(0.1)
            self.mu.bias.data.mul_(0.0)
        
        # self.apply(weights_init_)
        
    def forward(self, state):
        feature = self.feature_extractor(state)
        action_mean = self.mu(feature)
        action_log_std = self.log_std if self.state_std_independent else self.log_std(feature)
        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return action_mean, action_log_std.exp()
        