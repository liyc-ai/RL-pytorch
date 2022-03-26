import os
import gym
import d4rl  # Import required to register environments
import torch
import numpy as np
from algo import ALGOS
from utils.config import read_config
from utils.logger import get_logger, get_writer

# for safefy
from torch.utils.backcompat import broadcast_warning, keepdim_warning
broadcast_warning.enabled = True
keepdim_warning.enabled = True

# some needed directories
log_dir = 'logs'
tb_dir = 'tb'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True)

state_normalizer = lambda x: x
logger = None

def eval(imitator, env_name, seed, eval_episodes):
    global state_normalizer, logger
    
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = imitator.select_action(state_normalizer(state), training=False)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    
    avg_reward /= eval_episodes
    logger.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    
    return avg_reward

def train_imitator(configs, seed):
    global state_normalizer, logger
    # load expert dataset
    env = gym.make('halfcheetah-expert-v2')
    configs['state_dim'] = env.observation_space.shape[0]
    configs['action_space'] = env.action_space
    configs['dataset'] = env.get_dataset()
    
    # fix all the seeds
    env.seed(seed)
    env.action_space.seed(seed)  # we may use env.action_space.sample(), especially at the warm start period of training
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # prepare training
    configs['device'] = torch.device(configs['device']) if torch.cuda.is_available() else torch.device('cpu')
    file_name = f"{configs['algo_name']}_{configs['env_name']}_{seed}"
    
    logger = get_logger(os.path.join(log_dir, file_name+'.log'))
    writer = get_writer(os.path.join(tb_dir, file_name))
    
    # init imitator
    imitator = ALGOS[configs['algo_name']](configs)
    writer.add_scalar('evaluation_averaged_return', eval(imitator, configs['env_name'], seed, 10), global_step=0)  # evaluate before update, to get baseline
    
    # train imitator
    imitator.learn(logger, writer)
    
    # insert the expert dataset into the replay buffer
    # print('Loading buffer!')
    # for i in range(N-1):
    #     obs = dataset['observations'][i]
    #     new_obs = dataset['observations'][i+1]
    #     action = dataset['actions'][i]
    #     reward = dataset['rewards'][i]
    #     done_bool = bool(dataset['terminals'][i])
    #     replay_buffer.add(obs, action, new_obs, reward, done_bool)
    
if __name__ == '__main__':
    # read configs
    configs = read_config()
    # train imitator
    train_imitator(configs, seed=0)
    
