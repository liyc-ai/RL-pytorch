import gym
import d4rl
import torch
from torch.distributions import Normal
from algo.base import BaseAgent
from algo.imitation.bc import BCAgent
from algo.rl.ppo import PPOAgent
from utils.buffer import ImitationReplayBuffer
from utils.env import ConvertActionWrapper
from utils.config import load_yml_config
from tqdm import tqdm

class BC2Agent(BaseAgent):
    def __init__(self, configs: dict):
        super().__init__(configs)
        
        self.env = ConvertActionWrapper(gym.make(configs.get("env_name")))
        self.env.seed(configs.get("seed"))
        
        self.expert_buffer = ImitationReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.device,
            configs.get("expert_buffer_size"),
        )
        # use ppo to train generator
        rl_config = load_yml_config("ppo.yml")
        (
            rl_config["state_dim"],
            rl_config["action_dim"],
            rl_config["action_high"],
        ) = (self.state_dim, self.action_dim, self.action_high)
        if configs.get("rl") is not None:
            rl_config = {
                **rl_config,
                **configs.get("rl"),
            }
        self.policy = PPOAgent(rl_config)
        self.gamma = self.policy.gamma
        
        self.train_bc_times = 1
        
        self.models = {
            **self.policy.models
        }
        
    def _rollout(self):
        done = True
        for i in range(self.policy.rollout_steps):
            if done:
                next_state = self.env.reset()
            state = next_state
            action, _ = self.policy(state, training=True, calcu_log_prob=False, keep_grad=False)
            next_state, reward, done, _ = self.env.step(action)
            real_done = done if i < self.env._max_episode_steps else False
            
            action = action.cpu()
            self.policy.replay_buffer.add(
                state, action, reward, next_state, float(real_done)
            )
        
    def __call__(self, state, training=False, calcu_log_prob=False, keep_grad=False):
        return self.policy(state, training=training, calcu_log_prob=calcu_log_prob, keep_grad=keep_grad)
        
    def _train_bc(self):
        bc_config = load_yml_config("bc.yml")
        (
            bc_config["state_dim"],
            bc_config["action_dim"],
            bc_config["action_high"],
        ) = (self.state_dim, self.action_dim, self.action_high)
        if self.configs.get("bc") is not None:
            bc_config = {
                **bc_config,
                **self.configs.get("bc")
            }
        self.bc = BCAgent(bc_config)
        # train bc
        print("Training BC...")
        for _ in tqdm(range(bc_config["max_iters"])):
            snapshot = self.bc.update_param(self.expert_buffer)
        print(f"Finish training BC! Final BC loss: {snapshot['loss']}")
            
    def _bc_reward(self, states, actions):
        with torch.no_grad():
            mu, std = self.policy.actor(states)
            dist = Normal(mu, std)
            log_prob = torch.sum(dist.log_prob(actions), axis=-1, keepdim=True)
            return log_prob.exp()
        
    def update_param(self):
        # train bc. and then fixed
        if self.train_bc_times > 0:
            self._train_bc()
            self.train_bc_times -= 1
        # sample trajectories
        self._rollout()
        
        (
            states,
            actions,
            _,
            next_states,
            not_dones,
        ) = self.policy.replay_buffer.sample()
        rewards = self._bc_reward(states, actions)
        
        snapshot = self.policy.update_param(
            states, actions, rewards, next_states, not_dones
        )
        self.policy.replay_buffer.clear()
        return snapshot
        
    def learn(self):
        return self.update_param()