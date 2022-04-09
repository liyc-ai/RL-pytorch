import os
import gym
import h5py
import numpy as np
from tqdm import tqdm
import json


def _get_reset_data():
    data = dict(
        observations=[],
        next_observations=[],
        actions=[],
        rewards=[],
        terminals=[],
        timeouts=[],
    )
    return data


def generate_expert_dataset(agent, env_name, seed, max_steps=int(1e6)):
    env = gym.make(env_name)
    env.seed(seed)
    dataset, traj_data = _get_reset_data(), _get_reset_data()
    print("Start to rollout...")
    t = 0
    obs = env.reset()
    while len(dataset["rewards"]) < max_steps:
        t += 1
        action = agent.select_action(obs, training=False)
        next_obs, reward, done, _ = env.step(action)
        timeout, terminal = False, False
        if t == env._max_episode_steps:
            timeout = True
        elif done:
            terminal = done
        # insert transition
        traj_data["observations"].append(obs)
        traj_data["actions"].append(action)
        traj_data["next_observations"].append(next_obs)
        traj_data["rewards"].append(reward)
        traj_data["terminals"].append(terminal)
        traj_data["timeouts"].append(timeout)

        obs = next_obs
        if terminal or timeout:
            obs = env.reset()
            t = 0
            for k in dataset:
                dataset[k].extend(traj_data[k])
            traj_data = _get_reset_data()

    dataset = dict(
        observations=np.array(dataset["observations"]).astype(np.float32),
        actions=np.array(dataset["actions"]).astype(np.float32),
        next_observations=np.array(dataset["next_observations"]).astype(np.float32),
        rewards=np.array(dataset["rewards"]).astype(np.float32),
        terminals=np.array(dataset["terminals"]).astype(np.bool),
        timeouts=np.array(dataset["timeouts"]).astype(np.bool),
    )
    for k in dataset:
        dataset[k] = dataset[k][:max_steps]  # clip the additional data
    # add env info, for learning
    dataset["env_info"] = [
        env.observation_space.shape[0],
        env.action_space.shape[0],
        float(env.action_space.high[0]),
    ]
    return dataset


def _get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def read_hdf5_dataset(data_file_path):
    dataset = dict()
    with h5py.File(data_file_path, "r") as dataset_file:
        for k in tqdm(_get_keys(dataset_file), desc="load datafile"):
            dataset[k] = dataset_file[k][:]
    return dataset


def split_dataset(dataset):
    """split dataset into trajectories, return a list of start index"""
    max_step = dataset["observations"].shape[0]
    timeout_idx = np.where(dataset["timeouts"] == True)[0] + 1
    real_done_idx = np.where(dataset["terminals"] == True)[0] + 1
    start_idx = sorted(
        [0]
        + timeout_idx[timeout_idx < max_step].tolist()
        + real_done_idx[real_done_idx < max_step].tolist()
        + [max_step]
    )
    traj_pair = list(zip(start_idx[:-1], start_idx[1:] - 1))
    return traj_pair


def get_trajectory(dataset, start_idx, end_idx):
    ...
