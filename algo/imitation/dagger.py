import gym
import d4rl
from algo.imitation.bc import BCAgent


class DAggerAgent(BCAgent):
    """Dataset Aggregation"""

    def __init__(self, configs):
        super().__init__(configs)

        self.env = gym.make(configs["env_name"])
        self.env.seed(configs["seed"])
        self.rollout_step = configs["rollout_step"]

    def rollout(self, expert):
        done = True
        for _ in range(self.rollout_step):
            if done:
                state = self.env.reset()
            action = expert(state)
            next_state, _, done, _ = self.env.step(action)
            self.replay_buffer.add(state, action)
            state = next_state
