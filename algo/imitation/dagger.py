import gym
import d4rl
from algo.imitation.bc import BCAgent


class DAggerAgent(BCAgent):
    """Dataset Aggregation"""

    def __init__(self, configs):
        super().__init__(configs)

        self.env = gym.make(configs["env_name"])
        self.env.seed(configs["seed"])
        self.rollout_steps = configs["rollout_steps"]

    def rollout(self, expert):
        done = True
        log_pi = 0.0  # we do not use log_pi in dagger
        for _ in range(self.rollout_steps):
            if done:
                next_state = self.env.reset()
            state = next_state
            action = self.actor(state, training=True)
            next_state, _, done, _ = self.env.step(action)

            action = expert(state, training=False)
            real_done = done if _ < self.env._max_episode_steps else False

            self.expert_buffer.add(state, action, log_pi, next_state, real_done)
