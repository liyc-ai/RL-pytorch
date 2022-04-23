import os
import random
import gym
import torch
import numpy as np
from torch.utils.backcompat import broadcast_warning, keepdim_warning
from utils.logger import get_logger, get_writer
from utils.config import write_config


def set_random_seed(seed, env=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)


def eval(
    agent,
    env_name,
    seed,
    logger,
    state_normalizer=lambda x: x,
    eval_episodes=10,
    seed_offset=100,
):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent(
                state_normalizer(state), training=False, calcu_log_prob=False
            )
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    logger.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")

    return avg_reward


def preprare_training(configs, result_dir="out"):
    # for safety
    broadcast_warning.enabled = True
    keepdim_warning.enabled = True

    # env info
    env = gym.make(configs["env_name"])
    configs["state_dim"] = env.observation_space.shape[0]
    configs["action_dim"] = env.action_space.shape[0]
    configs["action_high"] = float(env.action_space.high[0])

    # fix all the seeds
    seed = configs["seed"]
    set_random_seed(seed, env)

    # result dir
    exp_name = f"{configs['algo_name']}_{configs['env_name']}_{seed}"
    exp_path = os.path.join(result_dir, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    logger = get_logger(os.path.join(exp_path, "log.log"))
    writer = get_writer(os.path.join(exp_path, "tb"))
    write_config(configs, os.path.join(exp_path, "config.yml"))  # for reproducibility

    return [configs, env, exp_path, logger, writer, seed]
