import os
import signal
from os.path import join
from typing import Callable

import gymnasium as gym
import hydra
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from rlplugs.drls.env import get_env_info, make_env, reset_env_fn
from rlplugs.logger import TBLogger
from rlplugs.net.ptu import (
    clean_cuda,
    save_torch_model,
    set_eval_mode,
    set_torch,
    set_train_mode,
)
from stable_baselines3.common.utils import set_random_seed

import rlpyt
from rlpyt import BaseRLAgent


@th.no_grad
def eval_policy(
    eval_env: gym.Env,
    reset_env_fn: Callable,
    policy: BaseRLAgent,
    seed: int,
    episodes=10,
):
    """Evaluate Policy"""
    set_eval_mode(policy.models)
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
                **{"action_space": eval_env.action_space},
            )
            state, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward
        avg_rewards.append(avg_reward)
    set_train_mode(policy.models)

    # average
    return np.mean(avg_rewards)


@hydra.main(config_path="../conf", config_name="train_agent", version_base="1.3.1")
def main(cfg: DictConfig):
    cfg.work_dir = os.getcwd()
    # prepare experiment
    set_torch()
    clean_cuda()
    set_random_seed(cfg.seed, True)

    # setup logger
    logger = TBLogger(args=OmegaConf.to_object(cfg), record_param=cfg.log.record_param)

    # setup environment
    train_env, eval_env = (make_env(cfg.env.id), make_env(cfg.env.id))
    OmegaConf.update(cfg, "env[info]", get_env_info(eval_env), merge=False)

    # create agent
    agent = rlpyt.make(cfg)

    # train agent
    def ctr_c_handler(_signum, _frame):
        """If the program was stopped by ctr+c, we will save the model before leaving"""
        logger.console.warning("The program is stopped...")
        logger.console.info(
            save_torch_model(agent.models, logger.ckpt_dir, "stopped_model.pt")
        )  # save model
        exit(1)

    signal.signal(signal.SIGINT, ctr_c_handler)

    agent.learn(train_env, eval_env, reset_env_fn, eval_policy, logger)

    # save model
    logger.console.info(
        save_torch_model(agent.models, logger.ckpt_dir, "final_model.pt")
    )


if __name__ == "__main__":
    main()
