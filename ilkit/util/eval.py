from typing import Callable

import gymnasium as gym
import numpy as np

from ilkit.algo.base import BasePolicy


def eval_policy(
    eval_env: gym.Env,
    reset_env_fn: Callable,
    policy: BasePolicy,
    seed: int,
    episodes=10,
):
    """Evaluate Policy
    """
    policy.eval()
    avg_rewards = []
    for _ in range(episodes):
        (state, _), terminated, truncated = reset_env_fn(eval_env, seed), False, False
        avg_reward = 0.0
        while not (terminated or truncated):
            action = policy.select_action(
                state,
                deterministic=True,
                keep_dtype_tensor=False,
                return_log_prob=False,
                **{"action_space": eval_env.action_space}
            )
            state, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward
        avg_rewards.append(avg_reward)
    policy.train()

    # average
    return np.mean(avg_rewards)
