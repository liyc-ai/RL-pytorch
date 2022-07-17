import numpy as np
import torch


class SimpleReplayBuffer:
    def __init__(self, state_dim, action_dim, device, buffer_size=int(1e6)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.buffer_size = buffer_size
        self.clear()

    def add(self, state, action, reward, next_state, done):
        # insert (s, a, r, s', done)
        if state.ndim == 1:
            self.state_buffer[self.ptr] = state
            self.action_buffer[self.ptr] = action
            self.reward_buffer[self.ptr] = reward
            self.next_state_buffer[self.ptr] = next_state
            self.not_done_buffer[self.ptr] = 1.0 - done

            # update pointer and size
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
        else:
            batch_num = state.shape[0]
            if self.ptr + batch_num > self.buffer_size:
                batch_out = (self.ptr + batch_num) % self.buffer_size
                batch_left = batch_num - batch_out
                self.state_buffer[self.ptr :] = state[:batch_left]
                self.action_buffer[self.ptr :] = action[:batch_left]
                self.reward_buffer[self.ptr :] = reward[:batch_left]
                self.next_state_buffer[self.ptr :] = next_state[:batch_left]
                self.not_done_buffer[self.ptr :] = 1.0 - done[:batch_left]

                self.state_buffer[:batch_out] = state[batch_left:]
                self.action_buffer[:batch_out] = action[batch_left:]
                self.reward_buffer[:batch_out] = reward[batch_left:]
                self.next_state_buffer[:batch_out] = next_state[batch_left:]
                self.not_done_buffer[:batch_out] = 1.0 - done[batch_left:]

                self.ptr = batch_out
                self.size = min(self.size + batch_num, self.buffer_size)
            else:
                self.state_buffer[self.ptr : self.ptr + batch_num] = state
                self.action_buffer[self.ptr : self.ptr + batch_num] = action
                self.reward_buffer[self.ptr : self.ptr + batch_num] = reward
                self.next_state_buffer[self.ptr : self.ptr + batch_num] = next_state
                self.not_done_buffer[self.ptr : self.ptr + batch_num] = 1.0 - done

                self.ptr = (self.ptr + batch_num) % self.buffer_size
                self.size = min(self.size + batch_num, self.buffer_size)

    def sample(self, batch_size=None):
        idx = list(range(self.size))
        if batch_size != None:
            idx = np.random.choice(idx, size=batch_size, replace=False)
        else:
            idx = np.array(idx)

        return (
            torch.FloatTensor(self.state_buffer[idx]).to(self.device),
            torch.FloatTensor(self.action_buffer[idx]).to(self.device),
            torch.FloatTensor(self.reward_buffer[idx]).to(self.device).unsqueeze(1),
            torch.FloatTensor(self.next_state_buffer[idx]).to(self.device),
            torch.FloatTensor(self.not_done_buffer[idx]).to(self.device).unsqueeze(1),
        )

    def clear(self):
        self.state_buffer = np.zeros((self.buffer_size, self.state_dim))
        self.action_buffer = np.zeros((self.buffer_size, self.action_dim))
        self.reward_buffer = np.zeros((self.buffer_size))
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim))
        self.not_done_buffer = np.zeros((self.buffer_size))
        self.ptr = 0
        self.size = 0


class ImitationReplayBuffer:
    def __init__(self, state_dim, action_dim, device, buffer_size=int(1e6)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.buffer_size = buffer_size
        self.clear()

    def add(self, state, action, log_pi, next_state, done):
        # insert (s, a, r, s', done)
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.log_pi_buffer[self.ptr] = log_pi
        self.next_state_buffer[self.ptr] = next_state
        self.not_done_buffer[self.ptr] = 1.0 - done

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
            torch.FloatTensor(self.state_buffer[idx]).to(self.device),
            torch.FloatTensor(self.action_buffer[idx]).to(self.device),
            torch.FloatTensor(self.log_pi_buffer[idx]).to(self.device),
            torch.FloatTensor(self.next_state_buffer[idx]).to(self.device),
            torch.FloatTensor(self.not_done_buffer[idx]).to(self.device),
        )

    def clear(self):
        self.state_buffer = np.zeros((self.buffer_size, self.state_dim))
        self.action_buffer = np.zeros((self.buffer_size, self.action_dim))
        self.log_pi_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim))
        self.not_done_buffer = np.zeros((self.buffer_size, 1))
        self.ptr = 0
        self.size = 0
