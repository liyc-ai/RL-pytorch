import os
import gym
import numpy as np
import h5py
from algo import ALGOS
from utils.config import parse_args, load_yml_config
from utils.data import generate_expert_dataset
from utils.exp import eval, preprare_training
from utils.transform import Normalizer


def train(configs, agent, env, logger, writer, seed, model_path):
    # init state normalizer
    if configs["norm_state"]:
        state_normalizer = Normalizer()
    else:
        state_normalizer = lambda x: x

    # evaluate before update, to get baseline
    writer.add_scalar(
        "evaluation_averaged_return",
        eval(agent, configs["env_name"], seed, logger, state_normalizer),
        global_step=0,
    )

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
            action = agent(state, training=True, calcu_log_prob=False)
        # 2. conduct action
        next_state, reward, done, _ = env.step(action)
        next_state = state_normalizer(next_state)
        # 3. update agent
        real_done = (
            done if episode_timesteps < env._max_episode_steps else False
        )  # during training, exceed the env's max steps does not really mean end
        snapshot = agent.learn(state, action, reward, next_state, float(real_done))
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
            avg_reward = eval(
                agent, configs["env_name"], seed, logger, state_normalizer
            )
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
    # train agent
    expert = train(*preprare_training(configs))
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
