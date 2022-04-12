import gym
import d4rl
import torch.nn as nn
from torch.distributions.normal import Normal
from algo.imitation.bc import BCAgent
from utils.exp import get_expert

class DAggerAgent(BCAgent):
    '''Dataset Aggregation
    '''
    def __init__(self, configs):
        super().__init__(configs)
        
        self._expert = get_expert(configs['expert_model_path'])
        self.env = gym.make(configs['env_name'])
        self.env.seed(configs['seed'])
        self.rollout_step = configs['rollout_step']
        
    def rollout(self):
        done = True
        for _ in range(self.rollout_step):
            if done:
                state = self.env.reset()
            action = self._expert(state)
            next_state, _, done, _ = self.env.step(action)
            self.replay_buffer.add(state, action)
            state = next_state