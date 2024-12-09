import os
import sys
from os.path import join
from typing import Callable, Dict, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch as th
from drlplugs.drls.env import get_env_info, make_env, reset_env_fn
from drlplugs.exp.prepare import set_random_seed
from drlplugs.logger import TBLogger
from drlplugs.net.ptu import load_torch_model, set_torch
from drlplugs.ospy.dataset import get_dataset_holder, save_dataset_to_h5
from omegaconf import DictConfig, OmegaConf

sys.path.append("./")

from src import BaseRLAgent, create_agent


@th.no_grad()
def _collect_demo(
    policy: BaseRLAgent,
    env: gym.Env,
    reset_env_fn: Callable,
    save_dir: str,
    save_name: str,
    n_traj: int = 10,
    n_step: int = 1000_000,
    with_log_prob: bool = False,
    seed: int = 0,
) -> Tuple[str, Dict[str, np.ndarray]]:
    """Collect dataset

    :param n_traj: Collect [n_trajs] trajectories
    :param n_step: Collect [max_steps] transitions
    """
    collected_steps, collected_n_traj = 0, 0
    _dataset = get_dataset_holder(with_log_prob)

    next_obs, _ = reset_env_fn(env, seed)
    while collected_n_traj < n_traj or collected_steps < n_step:
        collected_steps += 1

        obs = next_obs
        if with_log_prob:
            action, log_prob = policy.select_action(
                obs, deterministic=True, return_log_prob=True
            )
        else:
            action = policy.select_action(
                obs, deterministic=True, return_log_prob=False
            )
        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

        # insert
        _dataset["observations"].append(obs)
        _dataset["actions"].append(action)
        _dataset["rewards"].append(reward)
        _dataset["next_observations"].append(next_obs)
        _dataset["terminals"].append(terminated)
        _dataset["timeouts"].append(truncated)

        if with_log_prob:
            _dataset["infos/action_log_probs"].append(log_prob)

        if terminated or truncated:
            next_obs, _ = reset_env_fn(env, seed)
            collected_n_traj += 1

    dataset = {}
    if with_log_prob:
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
    save_dataset_to_h5(dataset, save_dir, save_name)

    return (
        f"Successfully save expert demonstration into {save_dir}/{save_name}.hdf5!",
        dataset,
    )


@hydra.main(config_path="./conf", config_name="collect_demo", version_base="1.3.1")
def main(cfg: DictConfig):
    cfg.work_dir = os.getcwd()
    # prepare experiment
    set_torch()
    set_random_seed(cfg.seed)

    # setup logger
    logger = TBLogger(args=OmegaConf.to_object(cfg), record_param=cfg.log.record_param)

    # setup environment
    env = make_env(cfg.env.id)
    OmegaConf.update(cfg, "env[info]", get_env_info(env), merge=False)

    # create agent
    agent = create_agent(cfg)
    logger.console.info(
        load_torch_model(agent.models, join(cfg.work_dir, cfg.expert_model_path))
    )

    # collect expert dataset
    logger.console.info(f"Collecting expert data on the environment {cfg.env.id}...")
    logger.console.info(
        _collect_demo(
            agent,
            env,
            reset_env_fn,
            cfg.demo.save_dir,
            cfg.demo.save_name,
            cfg.demo.n_traj,
            cfg.demo.n_step,
            cfg.demo.with_log_prob,
            cfg.seed,
        )[0]
    )


if __name__ == "__main__":
    main()
