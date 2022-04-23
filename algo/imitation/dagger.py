import gym
import d4rl
import torch
from algo.imitation.bc import BCAgent
from utils.env import ConvertActionWrapper


class DAggerAgent(BCAgent):
    """Dataset Aggregation"""

    def __init__(self, configs: dict):
        super().__init__(configs)

        self.env = ConvertActionWrapper(gym.make(configs.get("env_name")))
        self.env.seed(configs.get("seed"))
        self.rollout_steps = configs.get("rollout_steps")

    def rollout(self, expert):
        done = True
        for _ in range(self.rollout_steps):
            if done:
                next_state = self.env.reset()
            state = next_state
            action, _ = self.actor(state, training=True, calcu_log_prob=False, keep_grad=False)
            next_state, _, done, _ = self.env.step(action)
            expert_action, exp_log_pi = expert(state, training=False, calcu_log_prob=False, keep_grad=False)
            real_done = done if _ < self.env._max_episode_steps else False
            
            expert_action = expert_action.cpu()
            self.expert_buffer.add(
                state, expert_action, exp_log_pi, next_state, real_done
            )
