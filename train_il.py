import os
import gym
import d4rl  # Import required to register environments
import torch
import numpy as np
from algo import ALGOS
from algo.base import BaseAgent
from utils.config import read_config, write_config
from utils.logger import get_logger, get_writer
from utils.data import read_hdf5_dataset, split_dataset, get_trajectory
from torch.utils.backcompat import broadcast_warning, keepdim_warning

state_normalizer = lambda x: x
logger = None


def eval(agent: BaseAgent, env_name, seed, eval_episodes):
    global state_normalizer, logger

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(state_normalizer(state), training=False)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    logger.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")

    return avg_reward


def train_imitator(configs, result_dir="out", data_dir="data/expert_data"):
    global state_normalizer, logger
    # fix all the seeds
    seed = configs["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # prepare training
    broadcast_warning.enabled = True
    keepdim_warning.enabled = True

    exp_name = f"{configs['algo_name']}_{configs['env_name']}_{seed}"
    exp_path = os.path.join(result_dir, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    configs["device"] = (
        torch.device(configs["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger = get_logger(os.path.join(exp_path, "log.log"))
    writer = get_writer(os.path.join(exp_path, "tb"))

    best_avg_reward = -np.inf

    write_config(configs, os.path.join(exp_path, "config.yml"))  # for reproducibility

    # load dataset
    if configs["use_d4rl"] and "d4rl_task_name" in configs:
        env = gym.make(configs["d4rl_task_name"])
        dataset = env.get_dataset()
        configs["state_dim"], configs["action_dim"], configs["action_high"] = (
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
        )
    else:
        assert "dataset_name" in configs, "Please specify dataset!"
        data_file_path = os.path.join(data_dir, configs["dataset_name"] + ".hdf5")
        dataset = read_hdf5_dataset(data_file_path)
        configs["state_dim"], configs["action_dim"], configs["action_high"] = (
            int(dataset["env_info"][0]),
            int(dataset["env_info"][1]),
            dataset["env_info"][2],
        )
        logger.info(f"Dataset loaded from {data_file_path}")

    # init imitator
    agent = ALGOS[configs["algo_name"]](configs)
    model_path = os.path.join(exp_path, "model.pt")
    if os.path.exists(model_path):
        agent.load_model(model_path)
        logger.info(f"Successfully load model: {model_path}")
    writer.add_scalar(
        "evaluation_averaged_return",
        eval(agent, configs["env_name"], seed, 10),
        global_step=0,
    )  # evaluate before update, to get baseline

    # train imitator
    traj_pair = split_dataset(dataset)
    expert_traj = np.random.choice(traj_pair, configs["max_traj_num"], replace=False)
    for i, (start_idx, end_idx) in enumerate(expert_traj):
        new_trajectory = get_trajectory(dataset, start_idx, end_idx)
        agent.learn(new_trajectory)
        avg_reward = eval(agent, configs["env_name"], seed, 10)
        writer.add_scalar(
            "evaluation_averaged_return", avg_reward, global_step=i + 1
        )  # evaluate before update, to get baseline
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save_model(model_path)


if __name__ == "__main__":
    # read configs
    configs = read_config(config_type="il")
    # train imitator
    train_imitator(configs)
