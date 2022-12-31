from typing import Callable

import numpy as np

from ilkit.algo import BasePolicy


def eval_policy(
    eval_env, 
    reset_env: Callable,
    policy: BasePolicy, 
    seed: int, 
    episodes=10
):
    """Evaluate Policy
    """
    policy.eval()
    avg_rewards = []
    for _ in range(episodes):
        (state, _), terminated, truncated = reset_env(eval_env, seed), False, False
        avg_reward = 0.0
        while not (terminated or truncated):
            action = policy.get_action(
                state,
                deterministic=True,
                keep_dtype_tensor=False,
                return_log_prob=False,
            )
            state, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward
        avg_rewards.append(avg_reward)
    policy.train()

    # average
    return np.mean(avg_rewards)
