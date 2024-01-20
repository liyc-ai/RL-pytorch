from typing import Dict

from omegaconf import DictConfig
from rlplugs.logger import LoggerType

from rlpyt._base import BaseRLAgent
from rlpyt._onlinerl import OnlineRLAgent
from rlpyt.ddpg import DDPGAgent
from rlpyt.ddqn import DDQNAgent
from rlpyt.dqn import DQNAgent
from rlpyt.dueldqn import DuelDQNAgent
from rlpyt.ppo import PPOAgent
from rlpyt.sac import SACAgent
from rlpyt.td3 import TD3Agent
from rlpyt.trpo import TRPOAgent

AGENTS: Dict[str, BaseRLAgent] = {
    "ddpg": DDPGAgent,
    "ddqn": DDQNAgent,
    "dqn": DQNAgent,
    "dueldqn": DuelDQNAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
    "td3": TD3Agent,
    "trpo": TRPOAgent,
}


def make(cfg: DictConfig, logger: LoggerType) -> BaseRLAgent:
    """To instantiate an agent"""

    def _get_agent(cfg: DictConfig, logger: LoggerType) -> BaseRLAgent:
        """For python annotations"""
        return AGENTS[cfg.agent.algo](cfg, logger)

    agent = _get_agent(cfg, logger)
    agent.setup_model()
    return agent


__version__ = "1.3.2"

__all__ = [
    "DDPGAgent",
    "DDQNAgent",
    "DQNAgent",
    "DuelDQNAgent",
    "PPOAgent",
    "SACAgent",
    "TD3Agent",
    "TRPOAgent",
    "BaseRLAgent",
    "OnlineRLAgent",
    "AGENTS",
]
