from typing import Dict

from ilkit.algo import BasePolicy
from ilkit.util.buffer import TransitionBuffer
from ilkit.util.data import DataHandler


class ILPolicy(BasePolicy):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def load_expert(self):
        # instantiate data handler
        self.data_handler = DataHandler(self.seed)

        # get expert dataset
        expert_info = self.cfg["expert_dataset"]
        dataset_file_path = self.parse_path(expert_info["dataset_file_path"])
        # TODO: Unify d4rl_env_id and env_id, need an update of D4RL
        if expert_info["d4rl_env_id"] != None and dataset_file_path != None:
            self.logger.warn(
                "User's own dataset and D4RL dataset are both specified, but we will ignore user's dataset"
            )
        expert_dataset = self.data_handler.parse_dataset(
            dataset_file_path, expert_info["d4rl_env_id"]
        )

        # load expert dataset into buffer
        expert_buffer_kwarg = {
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "action_dtype": self.action_dtype,
            "device": self.device,
            "buffer_size": self.algo_config["expert"]["buffer_size"],
        }
        self.expert_buffer = TransitionBuffer(**expert_buffer_kwarg)
        self.data_handler.load_dataset_to_buffer(
            expert_dataset, self.expert_buffer, expert_info["n_expert_traj"]
        )
