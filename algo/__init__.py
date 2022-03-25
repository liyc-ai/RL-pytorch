from .rl.trpo import TRPOAgent
from .rl.ppo import PPOAgent
from .rl.sac import SACAgent

ALGOS = {
    'trpo': TRPOAgent,
    'ppo': PPOAgent,
    'sac': SACAgent,
}