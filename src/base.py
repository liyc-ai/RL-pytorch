from abc import ABC, abstractmethod
from os.path import join
from typing import Callable, Dict, Union

import gymnasium as gym
import numpy as np
import torch as th
from omegaconf import DictConfig
from torch import nn, optim
from tqdm import trange

from src.utils.drls.buffer import TransitionBuffer
from src.utils.logger import TBLogger
from src.utils.net.ptu import save_torch_model, tensor2ndarray


class BaseRLAgent(ABC):
    """Base for RL"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # hyper-param
        self.work_dir = cfg.work_dir
        self.device = th.device(cfg.device)
        self.seed = cfg.seed
        self.batch_size = self.cfg.agent.batch_size
        self.gamma = self.cfg.agent.gamma

        # bind env
        self.state_shape = tuple(cfg.env.info.state_shape)
        self.action_shape = tuple(cfg.env.info.action_shape)
        self.action_dtype = cfg.env.info.action_dtype

        # buffer
        buffer_kwarg = {
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "action_dtype": self.action_dtype,
            "device": self.device,
            "buffer_size": self.cfg.agent.buffer_size,
        }

        self.trans_buffer = TransitionBuffer(**buffer_kwarg)

        # models
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
        return_log_prob: bool,
        **kwarg,
    ) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Dict:
        """Provide the algorithm details for updating parameters"""
        raise NotImplementedError

    def learn(
        self,
        train_env: gym.Env,
        eval_env: gym.Env,
        reset_env_fn: Callable,
        eval_policy: Callable,
        logger: TBLogger,
    ):
        train_return = 0
        best_return = -float("inf")
        train_steps = self.cfg.train.max_steps
        eval_interval = self.cfg.train.eval_interval

        # start training
        if self.cfg.log.console_output:
            progress_f = None
            progress_bar = trange(train_steps)
        else:
            progress_f = open(join(logger.exp_dir, "progress.txt"), "w")
            progress_bar = trange(train_steps, file=progress_f)
        next_state, _ = reset_env_fn(train_env, self.seed)
        for t in progress_bar:
            state = next_state
            if "warmup_steps" in self.cfg.agent and t < self.cfg.agent.warmup_steps:
                action = train_env.action_space.sample()
            else:
                action = self.select_action(
                    state,
                    deterministic=False,
                    return_log_prob=False,
                    action_space=train_env.action_space,
                )
                action = tensor2ndarray((action,))[0]
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            train_return += reward

            # insert transition into buffer
            self.trans_buffer.insert_transition(
                state, action, next_state, reward, float(terminated)
            )

            # update policy
            logger.add_stats(self.update(), t)

            # whether this episode ends
            if terminated or truncated:
                logger.add_stats({"return/train": train_return}, t)
                next_state, _ = reset_env_fn(train_env, self.seed)
                train_return = 0

            # evaluate
            if (t + 1) % eval_interval == 0:
                eval_return = eval_policy(eval_env, reset_env_fn, self, self.seed)
                logger.add_stats({"return/eval": eval_return}, t)

                if eval_return > best_return:
                    logger.console.info(
                        f"Step {t}: get new best return: {eval_return}!"
                    )
                    save_torch_model(self.models, logger.ckpt_dir, "best_model")
                    best_return = eval_return

        if progress_f is not None:
            progress_f.close()
