from algo.imitation.base import BaseImitator

class BCImitator(BaseImitator):
    """Behavioral Cloning
    """
    def __init__(self, configs):
        super().__init__(configs)
      
    def select_action(self):
        ...
        
    def learn(self):
        ...
        