__version__ = "1.0.0"

from typing import Any, Dict

from mllogger import TBLogger

from rlbase.algo import *

# Reinforcement Learning
AGENTS: Dict[str, BasePolicy] = {
    "ddpg": DDPG,
    "ddqn": DDQN,
    "dqn": DQN,
    "dueldqn": DuelDQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "trpo": TRPO,
}


def _get_agent(cfg: Dict, logger: TBLogger) -> BasePolicy:
    return AGENTS[cfg["agent"]["algo"]](cfg, logger)


def make(cfg: Dict[str, Any], logger: TBLogger) -> BasePolicy:
    agent = _get_agent(cfg, logger)
    agent.setup_model()
    return agent
