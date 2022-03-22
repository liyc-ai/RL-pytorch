from .rl.trpo import TRPOAgent
from .rl.ppo import PPOAgent

ALGOS = {
    'trpo': TRPOAgent,
    'ppo': PPOAgent,
}