from typing import Callable, Dict

import gymnasium as gym
import numpy as np
import torch as th
from rlplugs.ospy.dataset import get_dataset_holder, save_dataset_to_h5

from rlpyt import BaseRLAgent


def eval_policy(
    eval_env: gym.Env,
    reset_env_fn: Callable,
    policy: BaseRLAgent,
    seed: int,
    episodes=10,
):
    """Evaluate Policy"""
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


@th.no_grad()
def collect_dataset(
    policy: BaseRLAgent,
    env: gym.Env,
    reset_env_fn: Callable,
    n_traj: int = 0,
    n_step: int = 0,
    save_dir: str = None,
    save_name: str = None,
    save_log_prob: bool = False,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Collect dataset

    :param n_traj: Collect [n_trajs] trajectories
    :param n_step: Collect [max_steps] transitions
    """
    collected_steps, collected_n_traj = 0, 0
    _dataset = get_dataset_holder(save_log_prob)

    next_obs, _ = reset_env_fn(env, seed)
    while collected_n_traj < n_traj or collected_steps < n_step:
        collected_steps += 1

        obs = next_obs
        if save_log_prob:
            action, log_prob = policy.select_action(
                obs, keep_dtype_tensor=False, deterministic=True, return_log_prob=True
            )
        else:
            action = policy.select_action(
                obs, keep_dtype_tensor=False, deterministic=True, return_log_prob=False
            )
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # insert
        _dataset["observations"].append(obs)
        _dataset["actions"].append(action)
        _dataset["rewards"].append(reward)
        _dataset["next_observations"].append(next_obs)
        _dataset["terminals"].append(terminated)
        _dataset["timeouts"].append(truncated)

        if save_log_prob:
            _dataset["infos/action_log_probs"].append(log_prob)

        if terminated or truncated:
            next_obs, _ = reset_env_fn(env, seed)
            collected_n_traj += 1

    dataset = {}
    if save_log_prob:
        dataset["infos/action_log_probs"] = np.array(
            _dataset["infos/action_log_probs"]
        ).astype(np.float64)

    dataset.update(
        dict(
            observations=np.array(_dataset["observations"]).astype(np.float64),
            actions=np.array(_dataset["actions"]).astype(np.float64),
            next_observations=np.array(_dataset["next_observations"]).astype(
                np.float64
            ),
            rewards=np.array(_dataset["rewards"]).astype(np.float64),
            terminals=np.array(_dataset["terminals"]).astype(np.bool_),
            timeouts=np.array(_dataset["timeouts"]).astype(np.bool_),
        )
    )

    # dump the saved dataset
    if all([save_dir, save_name]):
        save_dataset_to_h5(dataset, save_dir, save_name)

    return dataset
