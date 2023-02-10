import random
from os.path import join
from typing import Callable, Dict

import gym
import h5py
import numpy as np
import torch as th
from RLA import logger
from tqdm import tqdm

from ilkit.algo.base import BasePolicy
from ilkit.util.buffer import TransitionBuffer


class DataHandler:
    """Collect or load expert demonstrations
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    @th.no_grad()
    def collect_demo(
        self,
        expert: BasePolicy,
        env: gym.Env,
        reset_env_fn: Callable,
        n_traj: int = 0,
        n_step: int = 0,
        save_dir: str = None,
        file_name: str = None,
        save_log_prob: bool = False,
    ):
        """Collect expert demos
        
        :param n_traj: Collect [n_trajs] trajectories
        :param n_step: Collect [max_steps] transitions
        """
        collected_steps, collected_n_traj = 0, 0
        expert_data = self._get_dataset_holder(save_log_prob)

        next_obs, _ = reset_env_fn(env, self.seed)
        while collected_n_traj < n_traj or collected_steps < n_step:
            collected_steps += 1

            obs = next_obs
            if save_log_prob:
                action, log_prob = expert.select_action(
                    obs,
                    keep_dtype_tensor=False,
                    deterministic=True,
                    return_log_prob=True,
                )
            else:
                action = expert.select_action(
                    obs,
                    keep_dtype_tensor=False,
                    deterministic=True,
                    return_log_prob=False,
                )
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # insert
            expert_data["observations"].append(obs)
            expert_data["actions"].append(action)
            expert_data["rewards"].append(reward)
            expert_data["next_observations"].append(next_obs)
            expert_data["terminals"].append(terminated)
            expert_data["timeouts"].append(truncated)

            if save_log_prob:
                expert_data["infos/action_log_probs"].append(log_prob)

            if terminated or truncated:
                next_obs, _ = reset_env_fn(env, self.seed)
                collected_n_traj += 1

        dataset = {}
        if save_log_prob:
            dataset["infos/action_log_probs"] = np.array(
                expert_data["infos/action_log_probs"]
            ).astype(np.float64)

        dataset.update(
            dict(
                observations=np.array(expert_data["observations"]).astype(np.float64),
                actions=np.array(expert_data["actions"]).astype(np.float64),
                next_observations=np.array(expert_data["next_observations"]).astype(
                    np.float64
                ),
                rewards=np.array(expert_data["rewards"]).astype(np.float64),
                terminals=np.array(expert_data["terminals"]).astype(np.bool_),
                timeouts=np.array(expert_data["timeouts"]).astype(np.bool_),
            )
        )

        # dump expert demos
        if all([save_dir, file_name]):
            self._save_dataset_to_h5(dataset, save_dir, file_name)

        return dataset

    def parse_dataset(
        self, dataset_file_path: str = None, d4rl_env_id: str = None
    ) -> Dict:
        if d4rl_env_id is not None:
            return self.load_d4rl_dataset(d4rl_env_id)
        elif dataset_file_path is not None:
            return self.load_own_dataset(dataset_file_path)
        else:
            raise ValueError("You must specify the expert dataset!")

    def load_own_dataset(self, dataset_file_path: str):
        """Load your own dataset from the specified data_path
        """
        dataset = dict()
        with h5py.File(dataset_file_path, "r") as dataset_file:
            for k in tqdm(self._get_h5_keys(dataset_file), desc="load datafile"):
                dataset[k] = dataset_file[k][:]
        return dataset

    def load_d4rl_dataset(self, env_id: str):
        import d4rl

        env = gym.make(env_id)
        return env.get_dataset()

    def split_dataset_into_trajs(self, dataset: Dict):
        """Split the [D4RL] style dataset into trajectories
        
        :return: the corresponding start index and end index of every trajectories
        """
        max_steps = dataset["observations"].shape[0]
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
        traj_pair = list(zip(start_idx[:-1], start_idx[1:]))
        return traj_pair

    def get_one_traj(self, dataset, start_idx, end_idx):
        """Return a trajectory in dataset, from start_idx to end_idx (not included).
        """
        return {
            "observations": dataset["observations"][start_idx:end_idx],
            "actions": dataset["actions"][start_idx:end_idx],
            "rewards": dataset["rewards"][start_idx:end_idx],
            "next_observations": dataset["next_observations"][start_idx:end_idx],
            "terminals": dataset["terminals"][start_idx:end_idx],
            # "infos/action_log_probs": dataset["infos/action_log_probs"][
            #     start_idx:end_idx
            # ],
        }

    def load_dataset_to_buffer(
        self, dataset: Dict, buffer: TransitionBuffer, n_traj: int = None
    ):
        """Load [n_traj] trajs into the buffer
        """
        if n_traj is None:  # randomly select [traj_num] trajectories
            buffer.insert_dataset(dataset)
        else:
            traj_pair = self.split_dataset_into_trajs(dataset)
            traj_pair = random.sample(traj_pair, n_traj)

            for (start_idx, end_idx) in traj_pair:
                new_traj = self.get_one_traj(dataset, start_idx, end_idx)
                buffer.insert_dataset(new_traj)

    def _get_dataset_holder(self, with_log_prob: bool):
        """To determine the portions of demos
        """
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

    def _save_dataset_to_h5(self, dataset: Dict, save_dir: str, file_name: str):
        """To dump dataset into .hdf5 file
        
        :param dataset: Dataset to be saved
        :param save_dir: To save the collected demos
        :param file_name: File name of the saved demos
        """
        save_path = join(save_dir, file_name + ".hdf5")
        hfile = h5py.File(save_path, "w")
        for key, value in dataset.items():
            hfile.create_dataset(key, data=value, compression="gzip")

    def _get_h5_keys(self, h5file: h5py.File):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys


def load_expert_dataset(
    seed: int,
    expert_buffer_param: Dict,
    n_expert_traj: int,
    dataset_file_path: str = None,
    d4rl_env_id: str = None,
) -> TransitionBuffer:
    """Load expert dataset into TransitionBuffer
    """
    # instantiate data handler
    expert_data_handler = DataHandler(seed)

    # get expert dataset
    if d4rl_env_id is not None and dataset_file_path is not None:
        logger.warn(
            "User's own dataset and D4RL dataset are both specified, but we will ignore user's dataset"
        )
    expert_dataset = expert_data_handler.parse_dataset(dataset_file_path, d4rl_env_id)
    expert_buffer = TransitionBuffer(**expert_buffer_param)
    expert_data_handler.load_dataset_to_buffer(
        expert_dataset, expert_buffer, n_expert_traj
    )
    return expert_buffer
