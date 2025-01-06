import random

import numpy as np
import torch as th


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.

    :param seed:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices
    th.manual_seed(seed)
