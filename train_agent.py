import signal
from typing import Callable

import gymnasium as gym
import hydra
import numpy as np
import torch as th
from emg.exp.prepare import set_random_seed
from emg.exp.tracking import Tracking
from emg.nn.ptu import (
    save_torch_model,
    set_eval_mode,
    set_torch,
    set_train_mode,
    tensor2ndarray,
)
from emg.utils.drl.env import get_env_info, make_env, reset_env_fn
from omegaconf import DictConfig, OmegaConf

from src import BaseRLAgent, create_agent


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
    returns = []
    for _ in range(episodes):
        (state, _), terminated, truncated = reset_env_fn(eval_env, seed), False, False
        return_ = 0.0
        while not (terminated or truncated):
            action = policy.select_action(
                state,
                deterministic=True,
                return_log_prob=False,
                **{"action_space": eval_env.action_space},
            )
            state, reward, terminated, truncated, _ = eval_env.step(
                tensor2ndarray((action,))[0]
            )
            return_ += reward
        returns.append(return_)
    set_train_mode(policy.models)

    # average
    return np.mean(returns)


@hydra.main(config_path="./conf", config_name="train_agent", version_base="1.3.2")
def main(cfg: DictConfig):
    # prepare experiment
    set_torch()
    set_random_seed(cfg.seed)

    # setup manager
    tracking = Tracking(config=cfg)

    # setup environment
    train_env, eval_env = (make_env(cfg.env.id), make_env(cfg.env.id))
    OmegaConf.update(cfg, "env[info]", get_env_info(eval_env), merge=False)

    # create agent
    agent = create_agent(cfg)

    # train agent
    def ctr_c_handler(_signum, _frame):
        """If the program was stopped by ctr+c, we will save the model before leaving"""
        tracking.print("The program is stopped...")
        tracking.print(
            save_torch_model(agent.models, tracking.ckpt_dir, "stopped_model")
        )  # save model
        exit(1)

    signal.signal(signal.SIGINT, ctr_c_handler)

    agent.learn(train_env, eval_env, reset_env_fn, eval_policy, tracking)

    # save model
    tracking.print(save_torch_model(agent.models, tracking.ckpt_dir, "final_model"))


if __name__ == "__main__":
    main()
