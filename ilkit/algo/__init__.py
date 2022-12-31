from abc import ABC, abstractmethod
from os.path import exists, isabs, join
from typing import Dict, Union

import numpy as np
import torch as th
from RLA import exp_manager, logger
from torch import nn, optim

# exp_manager.new_saver()
# exp_manager.save_checkpoint()


class BasePolicy(ABC):
    """Base for RL and IL
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.algo_config = cfg["agent"]  # configs of algorithms
        
        # hyper-param
        self.root_dir = cfg["root_dir"]
        self.device = th.device(cfg["device"] if th.cuda.is_available() else "cpu")
        self.seed = cfg["seed"]
        
        # bind env
        env_info = cfg["env"]
        
        self.train_env = env_info["train"]
        self.eval_env = env_info["eval"]
        self.reset_env = env_info["reset_env"]
        
        self.state_shape = env_info["state_shape"]
        self.action_shape = env_info["action_shape"]
        self.action_dtype = env_info["action_dtype"]

        # experiment management
        self.logger = logger
        self.models: Dict[str, Union[nn.Module, optim.Optimizer, th.Tensor]] = dict()
        

    # -------- Initialization ---------
    @abstractmethod
    def init_param(self):
        raise NotImplementedError

    @abstractmethod
    def init_component(self):
        raise NotImplementedError

    # ------------- Logger ------------
    def setup_rla(self):
        log_info = self.cfg["log"]
        rla_config_file = self.parse_path(log_info["rla_config"])
        if log_info["setup_log"]:
            self.exp_manager = exp_manager
            self.exp_manager.configure(
                task_table_name=log_info["task_name"],
                private_config_path=rla_config_file,
                data_root=self.parse_path(log_info["log_dir"]),
                run_file=log_info["backup_file"],
                code_root=self.root_dir,
            )
            self.exp_manager.set_hyper_param(**self.cfg)
            self.exp_manager.add_record_param(log_info["record_param"])
            self.exp_manager.log_files_gen()
            self.exp_manager.print_args()

            # for logging
            self.results_dir = self.exp_manager.results_dir
            self.checkpoint_dir = self.exp_manager.checkpoint_dir
        else:
            self.logger.warn("We do not set up experiment manager!")

    @property
    def global_t(self):
        return self.exp_manager.time_step_holder.get_time()

    # --------  Interaction  ----------
    @abstractmethod
    def get_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg,
    ) -> Union[np.ndarray, th.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> Dict:
        """Provide the algorithm details for updating parameters
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self):
        raise NotImplementedError

    # ----------- Model ------------
    def eval(self):
        """Turn on eval mode
        """
        for model in self.models:
            if isinstance(model, nn.Module):
                self.models[model].eval()

    def train(self):
        """Turn on train mode
        """
        for model in self.models:
            if isinstance(model, nn.Module):
                self.models[model].train()

    def save_model(self, model_path: str):
        """Save model to pre-specified path
        
        Note: Currently, only th.Tensor and th.nn.Module are supported.
        """
        model_path = self.parse_path(model_path)

        state_dicts = {}
        for model_name in self.models:
            if isinstance(self.models[model_name], th.Tensor):
                state_dicts[model_name] = {model_name: self.models[model_name]}
            else:
                state_dicts[model_name] = self.models[model_name].state_dict()
        th.save(state_dicts, model_path)
        self.logger.info(f"Successfully save model to {model_path}!")

    def load_model(self, model_path):
        """Load model from pre-specified path
        
        Note: Currently, only th.Tensor and th.nn.Module are supported.
        """
        model_path = self.parse_path(model_path)
        if not exists(model_path):
            self.logger.warn(
                "No model to load, the model parameters are randomly initialized."
            )
            return
        state_dicts = th.load(model_path)
        for model_name in self.models:
            if isinstance(self.models[model_name], th.Tensor):
                self.models[model_name] = state_dicts[model_name][model_name]
            else:
                self.models[model_name].load_state_dict(state_dicts[model_name])
        self.logger.info(f"Successfully load model from {model_path}!")

    # ------------ Utilities ---------------
    def parse_path(self, path: str):
        """Convert relative path to absolute path
        """
        if path is not None and path != "":
            if not isabs(path):
                path = join(self.root_dir, path)
        return path
