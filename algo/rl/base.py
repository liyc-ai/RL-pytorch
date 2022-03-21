from abc import ABCMeta, abstractmethod

class BaseAgent(metaclass=ABCMeta):
    """Base agent class for RL
    """
    def __init__(self, configs):
        self.configs = configs
        
        self.state_dim = configs['state_dim']
        
        self.action_dim  = configs['action_space'].shape[0]
        self.action_high = float(configs['action_space'].high[0])
        self.action_space = configs['action_space']
        
        self.gamma = configs['gamma']
        self.device = configs['device']
        
    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def update(self):
        pass