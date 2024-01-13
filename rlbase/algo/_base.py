from abc import ABC, abstractmethod
from typing import Callable, Dict, Union

import gymnasium as gym
import numpy as np
import torch as th
from mllogger import TBLogger, WBLogger
from torch import nn, optim


class BasePolicy(ABC):
    """Base for RL"""

    def __init__(self, cfg: Dict, logger: Union[TBLogger, WBLogger]):
        self.cfg = cfg
        self.algo_cfg = cfg["agent"]  # configs of algorithms

        # hyper-param
        self.work_dir = cfg["work_dir"]
        self.device = th.device(cfg["device"] if th.cuda.is_available() else "cpu")
        self.seed = cfg["seed"]

        # bind env
        env_info = cfg["env"]

        self.state_shape = env_info["state_shape"]
        self.action_shape = env_info["action_shape"]
        self.action_dtype = env_info["action_dtype"]

        # experiment management
        self.logger = logger
        self.models: Dict[str, Union[nn.Module, optim.Optimizer, th.Tensor]] = dict()

    # -------- Initialization ---------
    @abstractmethod
    def setup_model(self):
        raise NotImplementedError

    # --------  Interaction  ----------
    @abstractmethod
    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg,
    ) -> Union[np.ndarray, th.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Dict:
        """Provide the algorithm details for updating parameters"""
        raise NotImplementedError

    @abstractmethod
    def learn(self, train_env: gym.Env, eval_env: gym.Env, reset_env_fn_fn: Callable):
        raise NotImplementedError

    def eval(self):
        """Turn on eval mode"""
        for model in self.models:
            if isinstance(model, nn.Module):
                self.models[model].eval()

    def train(self):
        """Turn on train mode"""
        for model in self.models:
            if isinstance(model, nn.Module):
                self.models[model].train()
