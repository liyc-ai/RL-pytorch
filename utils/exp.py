import torch
import numpy as np
import random


def set_random_seed(seed, env=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
