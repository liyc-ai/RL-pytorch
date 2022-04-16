import gym
import d4rl
import torch
from algo.imitation.bc import BCAgent


class DAggerAgent(BCAgent):
    """Dataset Aggregation"""

    def __init__(self, configs):
        super().__init__(configs)

        self.env = gym.make(configs.get("env_name"))
        self.env.seed(configs.get("seed"))
        self.rollout_steps = configs.get("rollout_steps")

    def rollout(self, expert):
        done = True
        for _ in range(self.rollout_steps):
            if done:
                next_state = self.env.reset()
            state = next_state
            with torch.no_grad():
                action, _ = self.actor(state, training=True, calcu_log_prob=False)
                action = action.cpu().data.numpy().flatten()
            next_state, _, done, _ = self.env.step(action)
            
            with torch.no_grad():  # in fact, dagger does not need log_pi
                expert_action, exp_log_pi = expert(state, training=False, calcu_log_prob=False)
            real_done = done if _ < self.env._max_episode_steps else False

            self.expert_buffer.add(state, expert_action, exp_log_pi, next_state, real_done)
