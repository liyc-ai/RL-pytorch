import os
from os.path import join
from typing import Dict, List

import h5py
import numpy as np
from tqdm import tqdm

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
                "You have to specify the max_episode_steps if no timeouts in dataset"
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
