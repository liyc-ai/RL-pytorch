__version__ = "0.0.1"

from typing import Any, Dict

from ilkit.algo.base import BasePolicy
from ilkit.algo.il import *
from ilkit.algo.rl import *
from mllogger import TBLogger

# Reinforcement Learning
RL_AGENTS: Dict[str, BasePolicy] = {
    "ddpg": DDPG,
    "ddqn": DDQN,
    "dqn": DQN,
    "dueldqn": DuelDQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "trpo": TRPO,
}

# Imitation Learning
IL_AGENTS: Dict[str, BasePolicy] = {
    "airl": AIRL,
    "bc_continuous": BCContinuous,
    "bc_discrete": BCDiscrete,
    "dagger_continuous": DAggerContinuous,
    "dagger_discrete": DAggerDiscrete,
    "gail": GAIL,
    # "dac": DAC,
    # "infogail": InfoGAIL,
    # "iq_learn_continuous": IQLearnContinuous,
    # "iq_learn_discrete": IQLearnDiscrete,
    # "value_dice": ValueDICE,
}

AGENTS: Dict[str, BasePolicy] = dict(RL_AGENTS, **IL_AGENTS)  # Merge two dicts


def _get_agent(cfg: Dict, logger: TBLogger) -> BasePolicy:
    return AGENTS[cfg["agent"]["algo"]](cfg, logger)


def make(cfg: Dict[str, Any], logger: TBLogger) -> BasePolicy:
    agent = _get_agent(cfg, logger)
    agent.setup_model()
    return agent
