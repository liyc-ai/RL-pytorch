from ilkit.algo.il.airl import AIRL
from ilkit.algo.il.bc import BCContinuous, BCDiscrete

# from ilkit.algo.il.dac import DAC
from ilkit.algo.il.dagger import DAggerContinuous, DAggerDiscrete
from ilkit.algo.il.gail import GAIL

# from ilkit.algo.il.infogail import InfoGAIL
# from ilkit.algo.il.iq_learn import IQLearnContinuous, IQLearnDiscrete
# from ilkit.algo.il.value_dice import ValueDICE

__all__ = [
    "AIRL",
    "BCContinuous",
    "BCDiscrete",
    "DAggerContinuous",
    "DAggerDiscrete",
    "GAIL",
    # "DAC",
    # "InfoGAIL",
    # "IQLearnContinuous",
    # "IQLearnDiscrete",
    # "ValueDICE",
]
