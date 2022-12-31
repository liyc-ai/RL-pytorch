__version__ = "0.0.1"

from typing import Dict, Union

from ilkit.algo import BasePolicy
from ilkit.algo.il import ILPolicy
# Imitation Learning
from ilkit.algo.il.airl import AIRL
from ilkit.algo.il.bc import BCContinuous, BCDiscrete
from ilkit.algo.il.dagger import DAggerContinuous, DAggerDiscrete
from ilkit.algo.il.gail import GAIL
from ilkit.algo.il.infogail import InfoGAIL
from ilkit.algo.il.iq_learn import IQLearnContinuous, IQLearnDiscrete
from ilkit.algo.il.value_dice import ValueDICE
from ilkit.algo.rl import OnlineRLPolicy
# Reinforcement Learning
from ilkit.algo.rl.ddpg import DDPG
from ilkit.algo.rl.ddqn import DDQN
from ilkit.algo.rl.dqn import DQN
from ilkit.algo.rl.dueldqn import DuelDQN
from ilkit.algo.rl.ppo import PPO
from ilkit.algo.rl.sac import SAC
from ilkit.algo.rl.td3 import TD3
from ilkit.algo.rl.trpo import TRPO


def get_agent(cfg: Dict) -> BasePolicy:
    AGENTS: Dict[str, BasePolicy] = {
        # Reinforcement Learning
        "ddpg": DDPG,
        "ddqn": DDQN,
        "dqn": DQN,
        "dueldqn": DuelDQN,
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
        "trpo": TRPO,
        # Imitation Learning
        "airl": AIRL,
        "bc_continuous": BCContinuous,
        "bc_discrete": BCDiscrete,
        "dagger_continuous": DAggerContinuous,
        "dagger_discrete": DAggerDiscrete,
        "gail": GAIL,
        "infogail": InfoGAIL,
        "iq_learn_continuous": IQLearnContinuous,
        "iq_learn_discrete": IQLearnDiscrete,
        "value_dice": ValueDICE,
    }
    return AGENTS[cfg["agent"]["algo"]](cfg)


def make(cfg: Dict) -> Union[BasePolicy, OnlineRLPolicy, ILPolicy]:
    agent = get_agent(cfg)
    agent.init_param()
    agent.init_component()
    return agent
