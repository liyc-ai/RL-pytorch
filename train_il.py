import os
import gym
import numpy as np
import random
from algo import ALGOS
from utils.config import parse_args, load_yml_config, write_config
from utils.logger import get_logger, get_writer
from utils.data import read_hdf5_dataset, split_dataset, load_expert_traj
from utils.exp import set_random_seed
from utils.env import add_env_info, ConvertActionWrapper
from torch.utils.backcompat import broadcast_warning, keepdim_warning
from train_expert import train

logger = None
eval_num = 0


def get_expert(configs):
    """Train expert if necessary"""
    expert_configs = load_yml_config(configs["expert_config"])
    env = gym.make(configs["env_name"])
    configs = add_env_info(configs, env=env)  # update env info
    if configs.get("expert_model") and os.path.exists(configs["expert_model"]):
        expert_configs = add_env_info(expert_configs, env=env)
        expert = ALGOS[configs["expert_name"]](expert_configs)
        expert.load_model(configs["expert_model"])
    else:
        expert = train(configs)
    return expert, configs


def eval(agent, env_name, seed, eval_episodes):
    global logger, eval_num

    eval_num += 1

    eval_env = ConvertActionWrapper(gym.make(env_name))
    eval_env.seed(seed + 100)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action, _ = agent(state, training=False, calcu_log_prob=False, keep_grad=False)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    logger.info(
        f"Evaluation times: {eval_num} Evaluation over {eval_episodes} episodes: {avg_reward:.3f}"
    )

    return avg_reward


def train_imitator(configs, result_dir="out", data_dir="data/expert_data"):
    global logger

    # fix all the seeds
    seed = configs["seed"]
    set_random_seed(seed)

    # prepare training
    broadcast_warning.enabled = True
    keepdim_warning.enabled = True

    if configs.get("use_d4rl") and configs.get("d4rl_task_name"):
        configs["env_name"] = configs["d4rl_task_name"]

    exp_name = f"{configs['algo_name']}_{configs['env_name']}_{seed}"
    exp_path = os.path.join(result_dir, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    logger = get_logger(os.path.join(exp_path, "log.log"))
    writer = get_writer(os.path.join(exp_path, "tb"))
    write_config(configs, os.path.join(exp_path, "config.yml"))  # for reproducibility

    # load dataset
    if configs["use_d4rl"] and configs.get("d4rl_task_name"):
        env = gym.make(configs["d4rl_task_name"])
        dataset = env.get_dataset()
        configs = add_env_info(configs, env=env)
    elif configs.get("dataset_name"):  # self generated dataset
        data_file_path = os.path.join(data_dir, configs["dataset_name"] + ".hdf5")
        dataset = read_hdf5_dataset(data_file_path)
        configs = add_env_info(configs, env_info=dataset["env_info"])
        logger.info(f"Dataset loaded from {data_file_path}")
    elif configs.get("query"):
        # for algo, which actively query expert during training
        expert, configs = get_expert(configs)

    # init agent
    agent = ALGOS[configs["algo_name"]](configs)
    model_path = os.path.join(exp_path, "model.pt")
    if configs["load_model"] and os.path.exists(model_path):
        agent.load_model(model_path)
        logger.info(f"Successfully load model: {model_path}")

    expert_traj_num = configs["expert_traj_num"]
    if expert_traj_num != 0:
        traj_pair = split_dataset(dataset)
        if len(traj_pair) < expert_traj_num:
            raise ValueError("Not enough expert trajectories!")
        expert_traj = random.sample(traj_pair, expert_traj_num)
        load_expert_traj(agent, dataset, expert_traj)

    # train agent
    writer.add_scalar(
        f"evaluation_averaged_return",
        eval(agent, configs["env_name"], seed, 10),
        global_step=0,
    )  # evaluate before update, to get baseline
    best_avg_reward = -np.inf
    for i in range(configs["max_iters"]):
        # 1. ask expert
        if "rollout_freq" in configs and i % configs["rollout_freq"] == 0:
            agent.rollout(expert)  # \beta = 1
            expert_traj_num += 1
            logger.info(
                f"Asking for new expert data, currently have {expert_traj_num} expert trajectories"
            )
        # 2. update
        snapshot = agent.learn()
        if snapshot != None:
            for key, value in snapshot.items():
                writer.add_scalar(key, value, global_step=i + 1)
        # 3. evaluate
        if (i + 1) % configs["eval_freq"] == 0:
            avg_reward = eval(agent, configs["env_name"], configs["seed"], 10)
            writer.add_scalar(
                f"evaluation_averaged_return",
                avg_reward,
                global_step=i + 1,
            )  # evaluate before update, to get baseline
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(model_path)


if __name__ == "__main__":
    # read configs
    args = parse_args()
    configs = load_yml_config(args.config)
    # train imitator
    train_imitator(configs)
