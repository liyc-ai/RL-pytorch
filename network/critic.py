import torch
import torch.nn as nn
from utils.net import build_mlp_extractor, weights_init_

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim=None, activation_fn=nn.Tanh, output_dim=1):
        super().__init__()
        self.action_dim = action_dim
        if action_dim != None:
            input_dim = state_dim + action_dim
        else:
            input_dim = state_dim
            
        feature_extractor = build_mlp_extractor(input_dim, hidden_size, activation_fn)
        value_head = nn.Linear(
            hidden_size[-1] if (hidden_size != None and len(hidden_size)>0) else input_dim,
            output_dim
        )

        value_head.weight.data.mul_(0.1)
        value_head.bias.data.mul_(0.0)
        
        # concat all the layer
        model = feature_extractor + [value_head]
        self.net = nn.Sequential(*model)
        
        # self.apply(weights_init_)
        
    def forward(self, state, action=None):
        if self.action_dim != None:
            assert action != None  
            x = torch.cat([state, action], dim=1)
        else:
            x = state
        return self.net(x) 
    