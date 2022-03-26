import os
import gym
import torch
import numpy as np
from algo import ALGOS
from utils.config import read_config
from utils.transform import Normalizer
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

def eval(agent, env_name, seed, eval_episodes):
    global state_normalizer, logger
    
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(state_normalizer(state), training=False)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    logger.info(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    
    return avg_reward

def train(configs, seed):
    global state_normalizer, logger
    # init environment
    env = gym.make(configs['env_name'])
    configs['state_dim'] = env.observation_space.shape[0]
    configs['action_space'] = env.action_space
    if configs['norm_state']:
        state_normalizer = Normalizer()
    
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
        
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0
    # init agent
    agent = ALGOS[configs['algo_name']](configs)
    writer.add_scalar('evaluation_averaged_return', eval(agent, configs['env_name'], seed, 10), global_step=0)  # evaluate before update, to get baseline
    
    # agent interects with environment
    next_state = state_normalizer(env.reset())
    for t in range(int(configs['max_timesteps'])):
        episode_timesteps += 1
        # 0. state transition
        state = next_state
        # 1. select action
        if 'start_timesteps' in configs.keys() and t < configs['start_timesteps']:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, training=True)
        # 2. conduct action
        next_state, reward, done, _ = env.step(action)
        next_state = state_normalizer(next_state)
        # 3. update agent
        real_done = done if episode_timesteps < env._max_episode_steps else False  # during training, exceed the env's max steps does not really mean end
        agent.update(state, action, next_state, reward, float(real_done))
        episode_reward += reward  # accumulate reward
        # 4. check env
        if done:
            next_state = state_normalizer(env.reset())
            logger.info(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            writer.add_scalar('training_return', episode_reward, global_step=t)
            # reset log variable
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        # 5. periodically evaluate learned policy
        if (t + 1) % configs['eval_freq'] == 0:
            writer.add_scalar('evaluation_averaged_return', eval(agent, configs['env_name'], seed, 10), global_step=t)

if __name__ == '__main__':
    # read configs
    configs = read_config()
    # train agent
    train(configs, seed=0)