import gym
import numpy as np

def _get_reset_data():
    data = dict(
        observations = [],
        next_observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = []
    )
    return data

def generate_expert_dataset(agent, env_name, seed, max_steps=int(1e6)):
    env = gym.make(env_name)
    env.seed(seed)
    dataset, traj_data = _get_reset_data(), _get_reset_data()
    print("Start to rollout...")
    t = 0 
    obs = env.reset()
    while len(dataset['rewards']) < max_steps:
        t += 1
        action = agent.select_action(obs, training=False)
        next_obs, reward, done, _ = env.step(action)
        timeout, terminal = False, False
        if t == max_steps:
            timeout = True
        elif done:
            terminal = done
        # insert transition
        traj_data['observations'].append(obs)
        traj_data['actions'].append(action)
        traj_data['next_observations'].append(next_obs)
        traj_data['rewards'].append(reward)
        traj_data['terminals'].append(terminal)
        traj_data['timeouts'].append(timeout)
        
        obs = next_obs
        if terminal or timeout:
            obs = env.reset()
            t = 0
            for k in dataset:
                dataset[k].extend(traj_data[k])
            traj_data = _get_reset_data()
       
    dataset = dict(
        observations=np.array(dataset['observations']).astype(np.float32),
        actions=np.array(dataset['actions']).astype(np.float32),
        next_observations=np.array(dataset['next_observations']).astype(np.float32),
        rewards=np.array(dataset['rewards']).astype(np.float32),
        terminals=np.array(dataset['terminals']).astype(np.bool),
        timeouts=np.array(dataset['timeouts']).astype(np.bool)
    )    
    for k in dataset:
        dataset[k] = dataset[k][:max_steps]  # clip the additional data
    # add env info, for learning   
    dataset['env_info'] = [env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0])]
    return dataset
    
def read_hdf5_dataset(data_file_path):
    ...