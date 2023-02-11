import time
from abc import ABC, abstractmethod
from os.path import exists, isabs, join
from re import A
from typing import Callable, Dict, Union

import gym
import numpy as np
import torch as th
from torch import nn, optim
from tqdm import trange

from ilkit.util.buffer import TransitionBuffer
from ilkit.util.logger import BaseLogger


class BasePolicy(ABC):
    """Base for RL and IL
    """

    def __init__(self, cfg: Dict, logger: BaseLogger):
        self.cfg = cfg
        self.algo_cfg = cfg["agent"]  # configs of algorithms

        # hyper-param
        self.work_dir = cfg["work_dir"]
        self.device = th.device(cfg["device"] if th.cuda.is_available() else "cpu")
        self.seed = cfg["seed"]

        # bind env
        env_info = cfg["env"]

        self.state_shape = env_info["state_shape"]
        self.action_shape = env_info["action_shape"]
        self.action_dtype = env_info["action_dtype"]

        # experiment management
        self.logger = logger
        self.models: Dict[str, Union[nn.Module, optim.Optimizer, th.Tensor]] = dict()

    # -------- Initialization ---------
    @abstractmethod
    def setup_model(self):
        raise NotImplementedError

    # --------  Interaction  ----------
    @abstractmethod
    def select_action(
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

    def learn(self, train_env: gym.Env, eval_env: gym.Env, reset_env_fn_fn: Callable):
        if self.cfg["hpf"]:
            self._nni_learn(train_env, eval_env, reset_env_fn_fn)
        else:
            self._no_nni_learn(train_env, eval_env, reset_env_fn_fn)
    
    @abstractmethod        
    def _nni_learn(self, train_env: gym.Env, eval_env: gym.Env, reset_env_fn_fn: Callable):
        raise NotImplementedError
    
    @abstractmethod
    def _no_nni_learn(self, train_env: gym.Env, eval_env: gym.Env, reset_env_fn_fn: Callable):
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
        if model_path is None:
            model_path = self.cfg["model_path"]

        state_dicts = {}
        for name, model in self.models.items():
            if isinstance(model, th.Tensor):
                state_dicts[name] = {name: model}
            else:
                state_dicts[name] = model.state_dict()
        th.save(state_dicts, model_path)

        self.logger.dump2log(f"Successfully save model to {model_path}!")

    def load_model(self, model_path: str = None):
        """Load model from pre-specified path
        Note: Currently, only th.Tensor and th.nn.Module are supported.
        """
        if model_path is None:
            model_path = self.cfg["model_path"]

        if not exists(model_path):
            self.logger.dump2log(
                "No model to load, the model parameters are randomly initialized."
            )
            return
        state_dicts = th.load(model_path)
        for name, model in self.models.items():
            if isinstance(model, th.Tensor):
                self.models[name] = state_dicts[name][name]
                self.__dict__[name].data = self.models[name].data
            else:
                model.load_state_dict(state_dicts[name])
        self.logger.dump2log(f"Successfully load model from {model_path}!")


class ILPolicy(BasePolicy):
    def __init__(self, cfg: Dict, logger: BaseLogger):
        super().__init__(cfg, logger)

    def load_expert(self):
        from ilkit.util.data import DataHandler

        # instantiate data handler
        self.data_handler = DataHandler(self.seed)

        # get expert dataset
        expert_info = self.cfg["expert_dataset"]
        dataset_file_path = expert_info["dataset_file_path"]
        if expert_info["d4rl_env_id"] is not None and dataset_file_path is not None:
            self.logger.dump2log(
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
            "buffer_size": -1,
        }
        self.expert_buffer = TransitionBuffer(**expert_buffer_kwarg)
        self.data_handler.load_dataset_to_buffer(
            expert_dataset, self.expert_buffer, expert_info["n_expert_traj"]
        )


class OnlineRLPolicy(BasePolicy):
    def __init__(self, cfg: Dict, logger: BaseLogger):
        super().__init__(cfg, logger)

        # hyper-param
        self.batch_size = self.algo_cfg["batch_size"]
        self.gamma = self.algo_cfg["gamma"]

        # buffer
        buffer_kwarg = {
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "action_dtype": self.action_dtype,
            "device": self.device,
            "buffer_size": self.algo_cfg["buffer_size"],
        }

        self.trans_buffer = TransitionBuffer(**buffer_kwarg)

    def _nni_learn(self, train_env: gym.Env, eval_env: gym.Env, reset_env_fn: Callable):
        import nni

        from ilkit.util.eval import eval_policy

        train_return = 0
        best_return = -float("inf")
        train_steps = self.cfg["train"]["max_steps"]
        eval_interval = self.cfg["train"]["eval_interval"]

        # start training
        next_state, _ = reset_env_fn(train_env, self.seed)
        for t in trange(train_steps):

            state = next_state
            if "warmup_steps" in self.algo_cfg and t < self.algo_cfg["warmup_steps"]:
                action = train_env.action_space.sample()
            else:
                action = self.select_action(
                    state,
                    keep_dtype_tensor=False,
                    deterministic=False,
                    return_log_prob=False,
                    **{"action_space": train_env.action_space},
                )
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            train_return += reward

            # insert transition into buffer
            self.trans_buffer.insert_transition(
                state, action, next_state, reward, terminated
            )

            # update policy
            self.update()

            # whether this episode ends
            if terminated or truncated:
                next_state, _ = reset_env_fn(train_env, self.seed)
                train_return = 0

            # evaluate
            if (t + 1) % eval_interval == 0:
                eval_return = eval_policy(eval_env, reset_env_fn, self, self.seed)
                nni.report_intermediate_result(eval_return)

                if eval_return > best_return:
                    best_return = eval_return

        nni.report_final_result(best_return)

    def _no_nni_learn(self, train_env: gym.Env, eval_env: gym.Env, reset_env_fn: Callable):
        from ilkit.util.eval import eval_policy

        if not self.cfg["train"]["learn"]:
            self.logger.dump2log("We did not learn anything!")
            return

        train_return = 0
        best_return = -float("inf")
        past_time = 0
        now_time = time.time()
        train_steps = self.cfg["train"]["max_steps"]
        eval_interval = self.cfg["train"]["eval_interval"]

        # start training
        next_state, info = reset_env_fn(train_env, self.seed)
        for t in trange(train_steps):
            last_time = now_time
            self.logger.set_global_t(t)

            state = next_state
            if "warmup_steps" in self.algo_cfg and t < self.algo_cfg["warmup_steps"]:
                action = train_env.action_space.sample()
            else:
                action = self.select_action(
                    state,
                    keep_dtype_tensor=False,
                    deterministic=False,
                    return_log_prob=False,
                    **{"action_space": train_env.action_space},
                )
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            train_return += reward

            # insert transition into buffer
            self.trans_buffer.insert_transition(
                state, action, next_state, reward, terminated
            )

            # update policy
            info = self.update()
            self.logger.logkvs(info)

            # whether this episode ends
            if terminated or truncated:
                self.logger.logkv("return/train", train_return)
                next_state, info = reset_env_fn(train_env, self.seed)
                train_return = 0

            # evaluate
            if (t + 1) % eval_interval == 0:
                eval_return = eval_policy(eval_env, reset_env_fn, self, self.seed)
                self.logger.logkv("return/eval", eval_return)

                if eval_return > best_return:
                    self.save_model(join(self.logger.checkpoint_dir, "best_model.pt"))
                    best_return = eval_return

            # update time
            now_time = time.time()
            one_step_time = now_time - last_time
            past_time += one_step_time
            if (t + 1) % self.cfg["log"]["print_time_interval"] == 0:
                remain_time = one_step_time * (train_steps - t - 1)
                self.logger.dump2log(
                    f"Run: {past_time/60} min, Remain: {remain_time/60} min"
                )

            self.logger.dumpkvs()
