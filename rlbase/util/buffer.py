import random
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import torch as th

# from stable_baselines3.common.buffers import ReplayBuffer


class BaseBuffer(ABC):
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        action_dtype: Union[np.int64, np.float32],
        device: Union[str, th.device],
        buffer_size: int = -1,
    ):
        """If buffer size is not specified, it will continually add new items in without removal of old items.
        """
        self.state_shape = state_shape
        self.action_shape = action_shape

        if action_dtype == "int":
            self.action_dtype = np.int64
        elif action_dtype == "float":
            self.action_dtype = np.float32
        else:
            raise ValueError("Unsupported action dtype!")

        self.device = device
        self.buffer_size = buffer_size if buffer_size != -1 else sys.maxsize
        self.buffers: List[th.Tensor] = []
        self.clear()

    @abstractmethod
    def init_buffer(self):
        raise NotImplementedError

    @abstractmethod
    def insert_transition(self):
        raise NotImplementedError

    @abstractmethod
    def insert_batch(self):
        raise NotImplementedError

    @abstractmethod
    def insert_dataset(self):
        raise NotImplementedError

    def load_dataset(self, dataset: Dict[str, np.ndarray], n_traj: int = None):
        """Load [n_traj] trajs into the buffer
        """
        from rlbase.util.data import get_one_traj, split_dataset_into_trajs

        if n_traj is None:
            self.insert_dataset(dataset)
        else:  # randomly select [traj_num] trajectories
            traj_pairs = split_dataset_into_trajs(dataset)
            traj_pair = random.sample(traj_pairs, n_traj)
            for (start_idx, end_idx) in traj_pair:
                new_traj = get_one_traj(dataset, start_idx, end_idx)
                self.insert_dataset(new_traj)

    def sample(self, batch_size: int = None, shuffle: bool = True):
        """Randomly sample items from the buffer.
        
        If batch_size is not provided, we will sample all the stored items.
        """
        idx = list(range(self.size))
        if shuffle:
            random.shuffle(idx)
        if batch_size is not None:
            idx = idx[:batch_size]

        return [buffer[idx] for buffer in self.buffers]

    def clear(self):
        self.init_buffer()
        self.buffers = [item.to(self.device) for item in self.buffers]
        self.ptr = 0
        self.size = 0
        self.total_size = 0  # Number of all the pushed items


class TransitionBuffer(BaseBuffer):
    """
    Transition buffer for single task
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        action_dtype: Union[np.int64, np.float32],
        device: Union[str, th.device],
        buffer_size: int = -1,
    ):
        super().__init__(state_shape, action_shape, action_dtype, device, buffer_size)

    def init_buffer(self):
        # Unlike some popular implementations,
        # we start with an empty buffer located in self.device (may be gpu).

        state_shape = (0,) + self.state_shape

        if self.action_dtype == np.int64:
            action_shape = (0, 1)
        else:
            action_shape = (0,) + self.action_shape

        self.buffers = [
            th.zeros(state_shape),  # state_buffer
            th.tensor(np.zeros(action_shape, dtype=self.action_dtype)),  # action_buffer
            th.zeros(state_shape),  # next_state_buffer
            th.zeros((0, 1)),  # reward_buffer
            th.zeros((0, 1)),  # done_buffer
        ]

    def insert_transition(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, int],
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ):
        # state
        state, next_state = (np.array(_, dtype=np.float32) for _ in [state, next_state])
        # action
        if self.action_dtype == np.int64:
            action = [action]
        action = np.array(action, dtype=self.action_dtype)
        # reward and done
        reward, done = (np.array([_], dtype=np.float32) for _ in [reward, done])

        new_transition = [state, action, next_state, reward, done]
        new_transition = [th.tensor(item).to(self.device) for item in new_transition]

        if self.total_size <= self.buffer_size:
            self.buffers = [
                th.cat((self.buffers[i], th.unsqueeze(new_transition[i], dim=0)), dim=0)
                for i in range(len(new_transition))
            ]
        else:
            for buffer, new_data in zip(self.buffers, new_transition):
                buffer[self.ptr] = new_data

        # update pointer and size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        self.total_size += 1

    def insert_batch(
        self,
        states: np.ndarray,
        actions: Union[np.ndarray, int],
        next_states: np.ndarray,
        rewards: float,
        dones: bool,
    ):
        """Insert a batch of transitions
        """
        for i in range(states.shape[0]):
            self.insert_transition(
                states[i], actions[i], next_states[i], rewards[i], dones[i]
            )

    def insert_dataset(self, dataset: Dict):
        """Insert dataset into the buffer
        """
        observations, actions, next_observations, rewards, terminals = (
            dataset["observations"],
            dataset["actions"],
            dataset["next_observations"],
            dataset["rewards"],
            dataset["terminals"],
        )  # we currently not consider the log_pis. But you can insert it with small modifications
        self.insert_batch(observations, actions, next_observations, rewards, terminals)

    def save_buffer(self, save_dir: str, file_name: str = None):
        from rlbase.util.data import save_dataset_to_h5

        buffer = {
            "observations": self.buffers[0].cpu().numpy(),
            "actions": self.buffers[1].cpu().numpy(),
            "next_observations": self.buffers[2].cpu().numpy(),
            "rewards": self.buffers[3].cpu().numpy(),
            "terminals": self.buffers[4].cpu().numpy(),
        }
        save_dataset_to_h5(
            buffer, save_dir, "buffer" if file_name is None else file_name
        )
