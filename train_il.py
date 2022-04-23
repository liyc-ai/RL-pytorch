import os
import gym
import numpy as np
import random
from algo import ALGOS
from utils.config import parse_args, load_yml_config
from utils.data import read_hdf5_dataset, split_dataset, load_expert_traj
from utils.exp import eval, preprare_training
from utils.env import add_env_info
from train_expert import train


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


def train_imitator(configs, env, exp_path, logger, writer, seed):
    # load dataset
    if configs.get("use_d4rl"):
        env = gym.make(configs["env_name"])
        dataset = env.get_dataset()
        configs = add_env_info(configs, env=env)
    elif configs.get("use_gen_data"):  # self generated dataset
        data_file_path = configs["dataset_path"]
        dataset = read_hdf5_dataset(data_file_path)
        configs = add_env_info(configs, env_info=dataset["env_info"])
    elif configs.get("query"):  # for dagger
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

    # evaluate before update, to get baseline
    writer.add_scalar(
        f"evaluation_averaged_return",
        eval(agent, configs["env_name"], seed, logger),
        global_step=0,
    )

    # train agent
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
            avg_reward = eval(agent, configs["env_name"], seed, logger)
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
    train_imitator(*preprare_training(configs))
