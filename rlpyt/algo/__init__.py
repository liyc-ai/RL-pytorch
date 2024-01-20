from rlpyt.algo._base import BaseRLAgent
from rlpyt.algo._onlinerl import OnlineRLAgent
from rlpyt.algo.ddpg import DDPG
from rlpyt.algo.ddqn import DDQN
from rlpyt.algo.dqn import DQN
from rlpyt.algo.dueldqn import DuelDQN
from rlpyt.algo.ppo import PPO
from rlpyt.algo.sac import SAC
from rlpyt.algo.td3 import TD3
from rlpyt.algo.trpo import TRPO
from typing import Dict

AGENTS: Dict[str, BaseRLAgent] = {
    "ddpg": DDPG,
    "ddqn": DDQN,
    "dqn": DQN,
    "dueldqn": DuelDQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "trpo": TRPO,
}

__all__ = [
    "DDPG",
    "DDQN",
    "DQN",
    "DuelDQN",
    "PPO",
    "SAC",
    "TD3",
    "TRPO",
    "BaseRLAgent",
    "OnlineRLAgent",
    "AGENTS"
]
