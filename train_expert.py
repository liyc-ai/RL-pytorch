import os
import gym
import numpy as np
import h5py
from algo import ALGOS
from utils.config import parse_args, load_yml_config, write_config
from utils.transform import Normalizer
from utils.logger import get_logger, get_writer
from utils.data import generate_expert_dataset
from utils.exp import set_random_seed
from torch.utils.backcompat import broadcast_warning, keepdim_warning

state_normalizer = lambda x: x
logger = None


def eval(agent, env_name, seed, eval_episodes):
    global state_normalizer, logger

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent(state_normalizer(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    logger.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")

    return avg_reward


def train(configs, result_dir="out"):
    global state_normalizer, logger
    # init environment
    env = gym.make(configs["env_name"])
    configs["state_dim"] = env.observation_space.shape[0]
    configs["action_dim"] = env.action_space.shape[0]
    configs["action_high"] = float(env.action_space.high[0])
    if configs["norm_state"]:
        state_normalizer = Normalizer()
    # fix all the seeds
    seed = configs["seed"]
    set_random_seed(seed, env)
    # prepare training
    broadcast_warning.enabled = True  # for safety
    keepdim_warning.enabled = True

    exp_name = f"{configs['algo_name']}_{configs['env_name']}_{seed}"
    exp_path = os.path.join(result_dir, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    logger = get_logger(os.path.join(exp_path, "log.log"))
    writer = get_writer(os.path.join(exp_path, "tb"))
    write_config(configs, os.path.join(exp_path, "config.yml"))  # for reproducibility
    # init agent
    agent = ALGOS[configs["algo_name"]](configs)
    model_path = os.path.join(exp_path, "model.pt")
    if configs.get("load_model") and os.path.exists(model_path):
        agent.load_model(model_path)
        logger.info(f"Successfully load model: {model_path}")

    writer.add_scalar(
        "evaluation_averaged_return",
        eval(agent, configs["env_name"], seed, 10),
        global_step=0,
    )  # evaluate before update, to get baseline
    # train agent
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    best_avg_reward = -np.inf

    next_state = state_normalizer(env.reset())
    for t in range(int(configs["max_timesteps"])):
        episode_timesteps += 1
        # 0. state transition
        state = next_state
        # 1. select action
        if (
            "start_timesteps" in configs.keys()
            and t < configs["start_timesteps"]
            and not os.path.exists(model_path)
        ):
            action = env.action_space.sample()
        else:
            action = agent(state, training=True)
        # 2. conduct action
        next_state, reward, done, _ = env.step(action)
        next_state = state_normalizer(next_state)
        # 3. update agent
        real_done = (
            done if episode_timesteps < env._max_episode_steps else False
        )  # during training, exceed the env's max steps does not really mean end
        snapshot = agent.learn(state, action, next_state, reward, float(real_done))
        if snapshot != None:
            for key, value in snapshot.items():
                writer.add_scalar(key, value, global_step=t + 1)
        episode_reward += reward  # accumulate reward
        # 4. check env
        if done:
            next_state = state_normalizer(env.reset())
            logger.info(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            writer.add_scalar("training_return", episode_reward, global_step=t)
            # reset log variable
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        # 5. periodically evaluate learned policy
        if (t + 1) % configs["eval_freq"] == 0:
            avg_reward = eval(agent, configs["env_name"], seed, 10)
            writer.add_scalar("evaluation_averaged_return", avg_reward, global_step=t)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(model_path)

    agent.load_model(model_path)  # use the best model
    return agent


if __name__ == "__main__":
    # read configs
    args = parse_args()
    configs = load_yml_config(args.config)
    configs["env_name"] = args.env_name
    # train agent
    expert = train(configs)
    if args.g:
        # generate expert dataset
        dataset = generate_expert_dataset(
            expert, configs["env_name"], configs["seed"] + 100
        )
        # save expert dataset, where the file name follows from d4rl
        data_dir = "data/expert_data"
        os.makedirs(data_dir, exist_ok=True)
        env_name, version = configs["env_name"].lower().split("-")
        data_file_path = os.path.join(
            data_dir, env_name + "_expert-" + version + ".hdf5"
        )

        hfile = h5py.File(data_file_path, "w")
        for k in dataset:
            hfile.create_dataset(k, data=dataset[k], compression="gzip")
        logger.info(f"Successfully save expert dataset to {data_file_path}")
