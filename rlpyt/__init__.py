from typing import Any, Dict

from rlplugs.logger import LoggerType

from rlpyt._base import BaseRLAgent
from rlpyt._onlinerl import OnlineRLAgent
from rlpyt.ddpg import DDPG
from rlpyt.ddqn import DDQN
from rlpyt.dqn import DQN
from rlpyt.dueldqn import DuelDQN
from rlpyt.ppo import PPO
from rlpyt.sac import SAC
from rlpyt.td3 import TD3
from rlpyt.trpo import TRPO

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


def make(cfg: Dict[str, Any], logger: LoggerType) -> BaseRLAgent:
    """To instantiate an agent"""

    def _get_agent(cfg: Dict, logger: LoggerType) -> BaseRLAgent:
        """For python annotations"""
        return AGENTS[cfg["agent"]["algo"]](cfg, logger)

    agent = _get_agent(cfg, logger)
    agent.setup_model()
    return agent


__version__ = "1.3.0"

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
    "AGENTS",
]
