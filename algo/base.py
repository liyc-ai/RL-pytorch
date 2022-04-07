import os
import torch
from abc import ABCMeta, abstractmethod
from utils.buffer import SimpleReplayBuffer

class BaseAgent(metaclass=ABCMeta):
    """Base agent class for RL
    """
    def __init__(self, configs):
        self.configs = configs
        
        self.state_dim = configs['state_dim']
        self.action_dim  = configs['action_dim']
        self.action_high = configs['action_high']
        
        self.gamma = configs['gamma']
        self.device = configs['device']
        self.replay_buffer = SimpleReplayBuffer(self.state_dim, self.action_dim, self.device, self.configs['buffer_size'])
        self.models = dict()
        
    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def learn(self):
        pass
    
    def _transform_action(self, action):
        return self.action_high*torch.tanh(action)  # squash and rescale output action
    
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model file not found: {}'.format(model_path))
        else:
            state_dicts = torch.load(model_path)
            for model in self.models:
                if isinstance(self.models[model], torch.Tensor):  # especially for sac, which has log_alpha to be loaded
                    self.models[model] = state_dicts[model][model]
                else:
                    self.models[model].load_state_dict(state_dicts[model])
        
    def save_model(self, model_path):
        if not self.models:
            raise ValueError("Models to be saved is \{\}!")
        state_dicts = {}
        for model in self.models:
            if isinstance(self.models[model], torch.Tensor):
                state_dicts[model] = {model: self.models[model]}
            else:
                state_dicts[model] = self.models[model].state_dict()
        torch.save(state_dicts, model_path)