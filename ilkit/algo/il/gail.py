import random
import time
from copy import deepcopy
from os.path import exists, join
from typing import Callable, Dict, Tuple, Union

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import BatchSampler
from tqdm import trange

from ilkit.algo.base import ILPolicy, OnlineRLPolicy
from ilkit.net.critic import MLPCritic
from ilkit.util.eval import eval_policy
from ilkit.util.logger import BaseLogger
from ilkit.util.ptu import gradient_descent


class GAIL(ILPolicy):
    """Generative Adversarial Imitation Learning (GAIL)
    """

    def __init__(self, cfg: Dict, logger: BaseLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        # hyper-param
        self.batch_size = self.algo_cfg["batch_size"]
        self.idx = list(range(self.algo_cfg["rollout_steps"]))

        # load expert dataset
        self.load_expert()

        # init discriminator
        self._init_disc()

        # init generator
        self._init_gen()

    def _init_disc(self):
        disc_kwarg = {
            "input_shape": (self.state_shape[0] + self.action_shape[0],),
            "output_shape": (1,),
            "net_arch": self.algo_cfg["discriminator"]["net_arch"],
            "activation_fn": getattr(
                nn, self.algo_cfg["discriminator"]["activation_fn"]
            ),
        }
        self.disc = MLPCritic(**disc_kwarg).to(self.device)
        self.disc_optim = getattr(optim, self.algo_cfg["discriminator"]["optimizer"])(
            self.disc.parameters(), self.algo_cfg["discriminator"]["lr"]
        )
        self.models.update({"disc": self.disc, "disc_optim": self.disc_optim})

    def _init_gen(self):
        from ilkit import make

        generator_cfg = deepcopy(self.cfg)
        generator_cfg["agent"] = OmegaConf.to_object(
            OmegaConf.load(self.algo_cfg["generator"])
        )
        self.generator: OnlineRLPolicy  = make(generator_cfg)

        for key, value in self.generator.models.items():
            self.models.update({"generator_" + key: value})

    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg,
    ) -> Union[Tuple[th.Tensor, th.Tensor], th.Tensor, np.ndarray]:
        return self.generator.select_action(
            state, deterministic, keep_dtype_tensor, return_log_prob, **kwarg
        )

    def _nni_learn(self, train_env: gym.Env, eval_env: gym.Env, reset_env_fn: Callable):
        import nni

        train_return = 0
        best_return = -float("inf")
        train_steps = self.cfg["train"]["max_steps"]
        eval_interval = self.cfg["train"]["eval_interval"]

        # start training
        next_state, _ = reset_env_fn(train_env, self.seed)
        for t in trange(train_steps):

            state = next_state
            action = self.select_action(
                state,
                keep_dtype_tensor=False,
                deterministic=False,
                return_log_prob=False,
            )
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            train_return += reward

            # insert transition into buffer
            self.generator.trans_buffer.insert_transition(
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
        next_state, _ = reset_env_fn(train_env, self.seed)
        for t in trange(train_steps):
            last_time = now_time
            self.logger.set_global_t(t)

            state = next_state
            action = self.select_action(
                state,
                keep_dtype_tensor=False,
                deterministic=False,
                return_log_prob=False,
            )
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            train_return += reward

            # insert transition into buffer
            self.generator.trans_buffer.insert_transition(
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

    def update(self):
        self.log_info = dict()

        if self.generator.trans_buffer.size >= self.algo_cfg["rollout_steps"]:
            # update discriminator
            self._update_disc()
            # prepare reward
            self._prepare_reward()
            # update generator
            self._update_gen()

        return self.log_info

    def _update_disc(self):
        for _ in range(self.algo_cfg["discriminator"]["n_update"]):
            random.shuffle(self.idx)
            batches = list(
                BatchSampler(self.idx, batch_size=self.batch_size, drop_last=False)
            )
            for batch in batches:
                # expert
                expert_states, expert_actions = self.expert_buffer.sample(
                    self.batch_size
                )[:2]
                d_expert = self.disc(expert_states, expert_actions)
                expert_loss = F.binary_cross_entropy_with_logits(
                    d_expert, th.ones_like(d_expert)
                )

                # imitator
                imitator_states = self.generator.trans_buffer.buffers[0][batch]
                imitator_actions = self.generator.trans_buffer.buffers[1][batch]
                d_imitator = self.disc(imitator_states, imitator_actions)
                imitator_loss = F.binary_cross_entropy_with_logits(
                    d_imitator, th.zeros_like(d_imitator)
                )

                loss = expert_loss + imitator_loss
                self.log_info.update(
                    {"loss/disc": gradient_descent(self.disc_optim, loss)}
                )

    def _update_gen(self):
        log_info = self.generator.update()
        for key, value in log_info.items():
            self.log_info["generator/" + key] = value

    def _prepare_reward(self):
        with th.no_grad():
            states, actions = self.generator.trans_buffer.buffers[:2]
            # -log(1-D(s,a)), trick from GAN
            rewards = -F.logsigmoid(-self.disc(states, actions))
            self.generator.trans_buffer.buffers[3] = rewards

    def load_model(self, model_path: str):
        if not exists(model_path):
            self.logger.dump2log(
                "No model to load, the model parameters are randomly initialized."
            )
            return
        state_dicts = th.load(model_path)
        for name, model in self.models.keys():
            if name.startswith("generator_"):
                _name = name[len("generator_") :]
                if isinstance(model, th.Tensor):
                    self.generator.models[_name] = state_dicts[name][name]
                    self.generator.__dict__[_name].data = self.generator.models[_name].data
                else:
                    self.generator.models[_name].load_state_dict(model)
            else:
                if isinstance(model, th.Tensor):
                    self.models[name] = state_dicts[name][name]
                    self.__dict__[name].data = self.models[name].data
                else:
                    self.models[name].load_state_dict(model)
        self.logger.dump2log(f"Successfully load model from {model_path}!")
