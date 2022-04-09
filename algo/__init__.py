from .rl.trpo import TRPOAgent
from .rl.ppo import PPOAgent
from .rl.sac import SACAgent
from .rl.td3 import TD3Agent
from .rl.ddpg import DDPGAgent
from .imitation.bc import BCAgent

ALGOS = {
    # Reinforcement Learning
    "trpo": TRPOAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
    "td3": TD3Agent,
    "ddpg": DDPGAgent,
    # Imitation Learning
    "bc": BCAgent,
}
