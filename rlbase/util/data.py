import os
from os.path import join
from typing import Callable, Dict, List

import gymnasium as gym
import h5py
import numpy as np
import torch as th
from tqdm import tqdm

from rlbase.algo import BasePolicy

# ================ Helpers =======================


def get_h5_keys(h5file: h5py.File) -> List[str]:
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def get_dataset_holder(with_log_prob: bool):
    """To determine the portions of demos"""
    dataset = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        timeouts=[],
    )
    if with_log_prob:
        dataset["infos/action_log_probs"] = []
    return dataset


# =============================  Collect  ==============================


@th.no_grad()
def collect_dataset(
    policy: BasePolicy,
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


# ================ Utility functions ================


def split_dataset_into_trajs(
    dataset: Dict[str, np.ndarray], max_episode_steps: int = None
):
    """Split the [D4RL] style dataset into trajectories

    :return: the corresponding start index and end index (not included) of every trajectories
    """
    max_steps = dataset["observations"].shape[0]
    if "timeouts" in dataset:
        timeout_idx = np.where(dataset["timeouts"] == True)[0] + 1
        terminal_idx = np.where(dataset["terminals"] == True)[0] + 1
        start_idx = sorted(
            set(
                [0]
                + timeout_idx[timeout_idx < max_steps].tolist()
                + terminal_idx[terminal_idx < max_steps].tolist()
                + [max_steps]
            )
        )
        traj_pairs = list(zip(start_idx[:-1], start_idx[1:]))
    else:
        if max_episode_steps is None:
            raise Exception(
                "You have the specify the max_episode_steps if no timeouts in dataset"
            )
        else:
            traj_pairs = []
            i = 0
            while i < max_steps:
                start_idx = i
                traj_len = 1
                while (traj_len <= max_episode_steps) and (i < max_steps):
                    i += 1
                    traj_len += 1
                    if dataset["terminals"][i - 1]:
                        break
                traj_pairs.append([start_idx, i])
    return traj_pairs


# =============================  Save  ==============================


def save_dataset_to_h5(dataset: Dict[str, np.ndarray], save_dir: str, file_name: str):
    """To dump dataset into .hdf5 file

    :param dataset: Dataset to be saved
    :param save_dir: To save the collected demos
    :param file_name: File name of the saved demos
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = join(save_dir, file_name + ".hdf5")
    hfile = h5py.File(save_path, "w")
    for key, value in dataset.items():
        hfile.create_dataset(key, data=value, compression="gzip")


# =============================  Get  ==============================


def get_one_traj(
    dataset: Dict[str, np.ndarray],
    start_idx: int,
    end_idx: int,
    with_log_prob: bool = False,
):
    """Return a trajectory in dataset, from start_idx to end_idx (not included)."""
    one_traj = {
        "observations": dataset["observations"][start_idx:end_idx],
        "actions": dataset["actions"][start_idx:end_idx],
        "rewards": dataset["rewards"][start_idx:end_idx],
        "next_observations": dataset["next_observations"][start_idx:end_idx],
        "terminals": dataset["terminals"][start_idx:end_idx],
    }
    if with_log_prob and "infos/action_log_probs" in dataset:
        one_traj.update(
            {
                "infos/action_log_probs": dataset["infos/action_log_probs"][
                    start_idx:end_idx
                ]
            }
        )
    return one_traj


def get_dataset(
    use_own_dataset: bool,
    own_dataset_path: str = None,
    d4rl_env_id: str = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    if use_own_dataset:
        assert (
            own_dataset_path is not None
        ), "To use your own dataset, you must fisrt specify your dataset path"
        dataset = dict()
        with h5py.File(own_dataset_path, "r") as dataset_file:
            for k in tqdm(get_h5_keys(dataset_file), desc="load datafile"):
                dataset[k] = dataset_file[k][:]
        return dataset
    else:
        import d4rl

        # d4rl is not compatible with gymnasium but only gym
        import gym

        env = gym.make(d4rl_env_id)
        import gymnasium as gym

        return env.get_dataset()
