import os
import torch
from abc import ABCMeta, abstractmethod
from utils.buffer import SimpleReplayBuffer
from utils.data import get_trajectory


class BaseAgent(metaclass=ABCMeta):
    def __init__(self, configs):
        self.configs = configs
        self.gamma = configs["gamma"]

        self.state_dim = int(configs["state_dim"])
        self.action_dim = int(configs["action_dim"])
        self.action_high = float(configs["action_high"])
        self.device = torch.device(
            configs["device"] if torch.cuda.is_available() else "cpu"
        )

        self.replay_buffer = SimpleReplayBuffer(
            self.state_dim, self.action_dim, self.device, self.configs["buffer_size"]
        )
        self.models = dict()

    @abstractmethod
    def __call__(self, state, training=False):
        pass

    @abstractmethod
    def learn(self):
        pass

    def rollout(self):
        pass

    def squash_action(self, action):
        return self.action_high * torch.tanh(action)  # squash and rescale output action

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found: {}".format(model_path))
        else:
            state_dicts = torch.load(model_path)
            for model in self.models:
                if isinstance(
                    self.models[model], torch.Tensor
                ):  # especially for sac, which has log_alpha to be loaded
                    self.models[model] = state_dicts[model][model]
                else:
                    self.models[model].load_state_dict(state_dicts[model])

    def save_model(self, model_path):
        if not self.models:
            raise ValueError("Models to be saved is \{\}!")
        state_dicts = {}
        for model in self.models:
            if isinstance(self.models[model], torch.Tensor):
                state_dicts[model] = {model: self.models[model]}
            else:
                state_dicts[model] = self.models[model].state_dict()
        torch.save(state_dicts, model_path)

    def load_expert_traj(self, dataset, expert_traj):
        for i, (start_idx, end_idx) in enumerate(expert_traj):
            new_traj = get_trajectory(dataset, start_idx, end_idx)
            observations, actions = new_traj["observations"], new_traj["actions"]
            traj_len = actions.shape[0]
            for i in range(traj_len):
                self.replay_buffer.add(observations[i], actions[i])
