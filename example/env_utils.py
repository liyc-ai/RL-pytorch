from typing import Dict

import gym
from gym.spaces import Box, Discrete


def _get_space_shape(obj: gym.Space):
    if isinstance(obj, Box):
        shape = obj.shape
    elif isinstance(obj, Discrete):
        shape = (obj.n,)
    else:
        raise TypeError("Currently only Box and Discrete are supported!")
    return shape

def get_env_info(env: gym.Env):
    state_shape = _get_space_shape(env.observation_space)
    action_shape = _get_space_shape(env.action_space)
    action_dtype = env.action_space.dtype
    return {
        "state_shape": state_shape,
        "action_shape": action_shape,
        "action_dtype": action_dtype,
    }


def make_env(env_id: str) -> gym.Env:
    """Currently we only support the below simple env style
    """
    try:
        env = gym.make(env_id)
    except:
        try:
            import d4rl
            env = gym.make(env_id)
        except:
            raise ValueError("Unsupported env id!")
    return env


def reset_env(env: gym.Env, seed: int) -> Dict:
    reset_info = env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return reset_info
