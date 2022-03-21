import torch
import numpy as np

class GAE():
    """Estimate Advantage using GAE (https://arxiv.org/abs/1506.02438)
    Ref:
    [1] https://nn.labml.ai/rl/ppo/gae.html
    [2] https://github.com/ikostrikov/pytorch-trpo
    """
    def __init__(self, gamma, lambda_, device):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.device = device
        
    def __call__(self, buffer_size, values, not_dones, rewards, last_value):
        Rs = torch.FloatTensor(np.zeros((buffer_size, 1))).to(self.device)  # reward to go R_t
        advantages = torch.FloatTensor(np.zeros((buffer_size, 1))).to(self.device)  # advantage

        last_advantage = torch.FloatTensor([0.]).to(self.device)
        last_return = last_value.clone()
        for t in reversed(range(buffer_size)):
            # calculate rewards-to-go reward
            Rs[t] = rewards[t] + self.gamma*last_return*not_dones[t]
            # delta and advantage
            delta = rewards[t] + self.gamma*last_value*not_dones[t] - values[t]
            advantages[t] = delta + self.gamma * self.lambda_*last_advantage*not_dones[t]
            # update pointer
            last_value = values[t].clone()
            last_advantage = advantages[t].clone()
            last_return = Rs[t].clone()
        return Rs, advantages 
        