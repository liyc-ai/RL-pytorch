from typing import Callable, Dict, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch as th
from emg import Manager
from emg.helper.drl.dataset import get_dataset_holder, save_dataset_to_h5
from emg.helper.drl.env import get_env_info, make_env, reset_env_fn
from emg.helper.exp.prepare import set_random_seed
from emg.helper.nn.ptu import load_torch_model, set_torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import BaseRLAgent, create_agent


@th.no_grad()
def _collect_demo(
    policy: BaseRLAgent,
    env: gym.Env,
    reset_env_fn: Callable,
    cfg: DictConfig,
    manager: Manager,
) -> Tuple[str, Dict[str, np.ndarray]]:
    """Collect dataset

    :param n_traj: Collect [n_trajs] trajectories
    :param n_step: Collect [max_steps] transitions
    """
    collected_steps, collected_n_traj = 0, 0
    _dataset = get_dataset_holder(cfg.demo.with_log_prob)

    for _ in tqdm(range(cfg.demo.n_traj)):
        next_obs, _ = reset_env_fn(env, cfg.seed)
        while True:
            obs = next_obs
            if cfg.demo.with_log_prob:
                action, log_prob = policy.select_action(
                    obs, deterministic=True, return_log_prob=True
                )
            else:
                action = policy.select_action(
                    obs, deterministic=True, return_log_prob=False
                )
            action = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # insert
            _dataset["observations"].append(obs)
            _dataset["actions"].append(action)
            _dataset["rewards"].append(reward)
            _dataset["next_observations"].append(next_obs)
            _dataset["terminals"].append(terminated)
            _dataset["timeouts"].append(truncated)

            if cfg.demo.with_log_prob:
                _dataset["infos/action_log_probs"].append(log_prob)

            if terminated or truncated:
                next_obs, _ = reset_env_fn(env, cfg.seed)
                break

    dataset = {}
    if cfg.demo.with_log_prob:
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
    save_dataset_to_h5(dataset, manager.log_dir, "expert_demo")

    return (
        f"Successfully save expert demonstration into {manager.log_dir}/expert_demo.hdf5!",
        dataset,
    )


@hydra.main(config_path="./conf", config_name="collect_demo", version_base="1.3.2")
def main(cfg: DictConfig):
    # prepare experiment
    set_torch()
    set_random_seed(cfg.seed)

    # setup manager
    manager = Manager(config=cfg)

    # setup environment
    env = make_env(cfg.env.id)
    OmegaConf.update(cfg, "env[info]", get_env_info(env), merge=False)

    # create agent
    agent = create_agent(cfg)
    manager.tracking.print(load_torch_model(agent.models, cfg.expert_model_path))

    # collect expert dataset
    manager.tracking.print(f"Collecting expert data on the environment {cfg.env.id}...")
    manager.tracking.print(_collect_demo(agent, env, reset_env_fn, cfg, manager)[0])


if __name__ == "__main__":
    main()
