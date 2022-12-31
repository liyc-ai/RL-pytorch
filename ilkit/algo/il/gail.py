import random
import time
from copy import deepcopy
from os.path import exists, join
from typing import Callable, Dict, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import BatchSampler

from ilkit.algo.il import ILPolicy
from ilkit.net.critic import MLPCritic
from ilkit.util.eval import eval_policy
from ilkit.util.ptu import gradient_descent


class GAIL(ILPolicy):
    """Generative Adversarial Imitation Learning (GAIL)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_param(self):
        self.batch_size = self.algo_config["batch_size"]
        self.idx = list(range(self.algo_config["rollout_steps"]))

    def init_component(self):
        # load expert dataset
        self.load_expert()

        # init discriminator
        self._init_disc()

        # init generator
        self._init_gen()

    def _init_disc(self):
        disc_kwarg = {
            "input_shape": (self.state_shape[0]+self.action_shape[0],),
            "output_shape": (1,),
            "net_arch": self.algo_config["discriminator"]["net_arch"],
            "activation_fn": getattr(nn, self.algo_config["discriminator"]["activation_fn"]),
        }
        self.disc = MLPCritic(**disc_kwarg).to(self.device)
        self.disc_optim = getattr(optim, self.algo_config["discriminator"]["optimizer"])(
            self.disc.parameters(), self.algo_config["discriminator"]["lr"]
        )
        self.models.update({"disc": self.disc, "disc_optim": self.disc_optim})

    def _init_gen(self):
        from ilkit import make

        generator_cfg = deepcopy(self.cfg)
        generator_cfg["agent"] = OmegaConf.to_object(OmegaConf.load(
            self.parse_path(self.algo_config["generator"])
        ))
        self.generator = make(generator_cfg)

        for key, value in self.generator.models.items():
            self.models.update({"generator_" + key: value})

    def get_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg,
    ):
        return self.generator.get_action(
            state, deterministic, keep_dtype_tensor, return_log_prob, **kwarg
        )

    def learn(self):
        if not self.cfg["train"]["learn"]:
            self.logger.warn("We did not learn anything!")
            return

        train_return = 0
        best_return = -float("inf")
        past_time = 0
        now_time = time.time()
        train_steps = self.cfg["train"]["max_steps"]
        eval_interval = self.cfg["train"]["eval_interval"]

        # start training
        next_state, info = self.reset_env(self.train_env, self.seed)
        for t in range(train_steps):
            last_time = now_time
            self.exp_manager.time_step_holder.set_time(t)

            state = next_state
            action = self.get_action(
                state,
                keep_dtype_tensor=False,
                deterministic=False,
                return_log_prob=False,
            )
            next_state, reward, terminated, truncated, _ = self.train_env.step(action)
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
                next_state, info = self.reset_env(self.train_env, self.seed)
                train_return = 0

            # evaluate
            if (t + 1) % eval_interval == 0:
                eval_return = eval_policy(self.eval_env, self.reset_env, self, self.seed)
                self.logger.logkv("return/eval", eval_return)

                # nni
                if self.cfg["hpo"]:
                    import nni
                    nni.report_intermediate_result(eval_return)

                if eval_return > best_return:
                    self.save_model(join(self.checkpoint_dir, "best_model.pt"))
                    best_return = eval_return

            # update time
            now_time = time.time()
            one_step_time = now_time - last_time
            past_time += one_step_time
            if (t + 1) % self.cfg["log"]["print_time_interval"] == 0:
                remain_time = one_step_time * (train_steps - t - 1)
                self.logger.info(
                    f"Run: {past_time/60} min, Remain: {remain_time/60} min"
                )

            self.logger.dumpkvs()

        if self.cfg["hpo"]:
            nni.report_final_result(best_return)

    def update(self):
        self.log_info = dict()

        if self.generator.trans_buffer.size >= self.algo_config["rollout_steps"]:
            # update discriminator
            self._update_disc()
            # prepare reward
            self._prepare_reward()
            # update generator
            self._update_gen()

        return self.log_info

    def _update_disc(self):
        for _ in range(self.algo_config["discriminator"]["n_update"]):
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
        model_path = self.parse_path(model_path)
        if not exists(model_path):
            self.logger.warn(
                "No model to load, the model parameters are randomly initialized."
            )
            return
        state_dicts = th.load(model_path)
        for model_name in self.models.keys():
            if model_name.startswith("generator_"):
                imitator_model_name = model_name[len("generator_") :]
                if isinstance(self.models[model_name], th.Tensor):
                    self.generator.models[imitator_model_name] = state_dicts[
                        model_name
                    ][model_name]
                else:
                    self.generator.models[imitator_model_name].load_state_dict(
                        state_dicts[model_name]
                    )
            else:
                if isinstance(self.models[model_name], th.Tensor):
                    self.models[model_name] = state_dicts[model_name][model_name]
                else:
                    self.models[model_name].load_state_dict(state_dicts[model_name])
        self.logger.info(f"Successfully load model from {model_path}!")
