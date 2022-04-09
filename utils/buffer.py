import numpy as np
import torch


class SimpleReplayBuffer:
    def __init__(self, state_dim, action_dim, device, buffer_size=int(1e6)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.buffer_size = buffer_size
        self.clear()

    def add(self, state, action, next_state, reward, done):
        # insert (s, a, r, s', done)
        self.state_buffer[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.action_buffer[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.next_state_buffer[self.ptr] = torch.FloatTensor(next_state).to(self.device)
        self.reward_buffer[self.ptr] = torch.as_tensor(reward).to(self.device)
        self.not_done_buffer[self.ptr] = torch.as_tensor(1.0 - done).to(self.device)

        # update pointer and size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size=None):
        idx = list(range(self.size))
        if batch_size != None:
            idx = np.random.choice(idx, size=batch_size, replace=False)
        else:
            idx = np.array(idx)

        return (
            self.state_buffer[idx],
            self.action_buffer[idx],
            self.next_state_buffer[idx],
            self.reward_buffer[idx],
            self.not_done_buffer[idx],
        )

    def clear(self):
        self.state_buffer = torch.zeros(
            (self.buffer_size, self.state_dim), dtype=torch.float32
        ).to(self.device)
        self.action_buffer = torch.zeros(
            (self.buffer_size, self.action_dim), dtype=torch.float32
        ).to(self.device)
        self.next_state_buffer = torch.zeros(
            (self.buffer_size, self.state_dim), dtype=torch.float32
        ).to(self.device)
        self.reward_buffer = torch.zeros((self.buffer_size, 1), dtype=torch.float32).to(
            self.device
        )
        self.not_done_buffer = torch.zeros(
            (self.buffer_size, 1), dtype=torch.float32
        ).to(self.device)
        self.ptr = 0  # position of the next inserted transition
        self.size = 0  # current num of inserted transitions
