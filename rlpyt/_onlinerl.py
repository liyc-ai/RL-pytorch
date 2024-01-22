from typing import Callable, Dict

import gymnasium as gym
from omegaconf import DictConfig
from rlplugs.drls.buffer import TransitionBuffer
from rlplugs.drls.env import get_env_info
from rlplugs.logger import LoggerType
from rlplugs.net.ptu import save_torch_model
from tqdm import trange

from rlpyt import BaseRLAgent
from rlpyt._helpers import eval_policy


class OnlineRLAgent(BaseRLAgent):
    def __init__(self, cfg: DictConfig, logger: LoggerType):
        super().__init__(cfg, logger)

        # hyper-param
        self.batch_size = self.cfg.agent.batch_size
        self.gamma = self.cfg.agent.gamma

        # buffer
        buffer_kwarg = {
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "action_dtype": self.action_dtype,
            "device": self.device,
            "buffer_size": self.cfg.agent.buffer_size,
        }

        self.trans_buffer = TransitionBuffer(**buffer_kwarg)

    def learn(self, train_env: gym.Env, eval_env: gym.Env, reset_env_fn: Callable):
        if not self.cfg.train.learn:
            self.logger.console.warning("We did not learn anything!")
            return

        train_return = 0
        best_return = -float("inf")
        train_steps = self.cfg.train.max_steps
        eval_interval = self.cfg.train.eval_interval

        # start training
        next_state, _ = reset_env_fn(train_env, self.seed)
        for t in trange(train_steps):
            state = next_state
            if "warmup_steps" in self.cfg.agent and t < self.cfg.agent.warmup_steps:
                action = train_env.action_space.sample()
            else:
                action = self.select_action(
                    state,
                    keep_dtype_tensor=False,
                    deterministic=False,
                    return_log_prob=False,
                    action_space=train_env.action_space,
                )
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            train_return += reward

            # insert transition into buffer
            self.trans_buffer.insert_transition(
                state, action, next_state, reward, terminated
            )

            # update policy
            self.logger.add_dict(self.update(), t)

            # whether this episode ends
            if terminated or truncated:
                self.logger.tb.add_scalar("return/train", train_return, t)
                next_state, _ = reset_env_fn(train_env, self.seed)
                train_return = 0

            # evaluate
            if (t + 1) % eval_interval == 0:
                eval_return = eval_policy(eval_env, reset_env_fn, self, self.seed)
                self.logger.tb.add_scalar("return/eval", eval_return, t)

                if eval_return > best_return:
                    self.logger.console.info(
                        f"Successfully save models into {save_torch_model(self.models, self.logger.ckpt_dir, 'best_model.pt')}"
                    )

                    best_return = eval_return
