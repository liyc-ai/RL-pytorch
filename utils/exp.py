import torch
import numpy as np
import random
import gym


def set_random_seed(seed, env=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)


def get_env_name(d4rl_task_name):
    env_info = d4rl_task_name.split("-")
    env_name, env_version = env_info[0], env_info[2]
    return env_name.title() + "-" + env_version


def add_env_info(configs, env=None, env_info=None):
    if env != None:
        configs["state_dim"], configs["action_dim"], configs["action_high"] = (
            env.observation_space.shape[0],
            env.action_space.shape[0],
            env.action_space.high[0],
        )
    elif env_info == None:
        configs = {**configs, **env_info}
    else:
        raise ValueError("env or env_info must be not None")

    return configs
