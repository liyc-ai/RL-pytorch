import importlib.metadata
from typing import Dict

from omegaconf import DictConfig

from .base import BaseRLAgent
from .ddpg import DDPGAgent
from .ddqn import DDQNAgent
from .dqn import DQNAgent
from .dueldqn import DuelDQNAgent
from .ppo import PPOAgent
from .sac import SACAgent
from .td3 import TD3Agent
from .trpo import TRPOAgent

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


def create_agent(cfg: DictConfig) -> BaseRLAgent:
    """To instantiate an agent"""

    def _get_agent(cfg: DictConfig) -> BaseRLAgent:
        """For python annotations"""
        return AGENTS[cfg.agent.algo](cfg)

    agent = _get_agent(cfg)
    agent.setup_model()
    return agent


__all__ = [
    DDPGAgent,
    DDQNAgent,
    DQNAgent,
    DuelDQNAgent,
    PPOAgent,
    SACAgent,
    TD3Agent,
    TRPOAgent,
    BaseRLAgent,
    AGENTS,
]
