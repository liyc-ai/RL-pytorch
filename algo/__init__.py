from .rl.trpo import TRPOAgent
from .rl.ppo import PPOAgent
from .rl.sac import SACAgent
from .rl.td3 import TD3Agent

ALGOS = {
    'trpo': TRPOAgent,
    'ppo': PPOAgent,
    'sac': SACAgent,
    'td3': TD3Agent,
}