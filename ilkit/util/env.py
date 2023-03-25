from typing import Dict

import gymnasium as gym
from gymnasium.spaces import Box, Discrete


def _get_space_info(obj: gym.Space):
    if isinstance(obj, Box):
        shape = obj.shape
        type_ = "float"
    elif isinstance(obj, Discrete):
        shape = (obj.n.item(),)
        type_ = "int"
    else:
        raise TypeError("Currently only Box and Discrete are supported!")
    return shape, type_


def get_env_info(env: gym.Env):
    state_shape, _ = _get_space_info(env.observation_space)
    action_shape, action_dtype = _get_space_info(env.action_space)

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


def reset_env_fn(env: gym.Env, seed: int) -> Dict:
    next_state, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return (next_state, info)
