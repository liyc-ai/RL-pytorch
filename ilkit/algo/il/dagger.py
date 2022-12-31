import time
from copy import deepcopy
from os.path import join
from typing import Dict, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn, optim
from torch.distributions.categorical import Categorical

from ilkit.algo.il.bc import BCContinuous
from ilkit.net.actor import MLPDeterministicActor, MLPGaussianActor
from ilkit.util.buffer import DAggerBuffer
from ilkit.util.eval import eval_policy
from ilkit.util.ptu import gradient_descent, tensor2ndarray


class DAggerContinuous(BCContinuous):
    """Dataset Aggregation (DAgger) for Continuous Control
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def load_expert(self):
        """expert could be anything, as long as it is able to give an action with state input
        """
        from ilkit import make

        expert_config = deepcopy(self.cfg)
        expert_config["agent"] = OmegaConf.to_object(OmegaConf.load(
            self.parse_path(self.algo_config["expert"]["config"])
        ))
        self.expert = make(expert_config)
        self.expert.load_model(self.algo_config["expert"]["model_path"])

    def init_component(self):
        # load expert
        self.load_expert()

        # buffer
        buffer_kwarg = {
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "action_dtype": self.action_dtype,
            "device": self.device,
            "buffer_size": self.algo_config["buffer_size"],
        }
        self.trans_buffer = DAggerBuffer(**buffer_kwarg)

        # actor
        actor_kwarg = {
            "state_shape": self.state_shape,
            "net_arch": self.algo_config["actor"]["net_arch"],
            "action_shape": self.action_shape,
            "state_std_independent": self.algo_config["actor"]["state_std_independent"],
            "activation_fn": getattr(nn, self.algo_config["actor"]["activation_fn"]),
        }
        self.actor = MLPGaussianActor(**actor_kwarg).to(self.device)
        self.actor_optim = getattr(optim, self.algo_config["actor"]["optimizer"])(
            self.actor.parameters(), self.algo_config["actor"]["lr"]
        )

        self.models.update({"actor": self.actor, "actor_optim": self.actor_optim})

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

            expert_action = self.expert.get_action(
                state,
                deterministic=True,
                keep_dtype_tensor=False,
                return_log_prob=False,
            )
            self.trans_buffer.insert_transition(state, expert_action)

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
        log_info = dict()
        if self.trans_buffer.size >= self.batch_size:
            states, actions = self.trans_buffer.sample(self.batch_size)
            log_info.update(
                {
                    "loss": gradient_descent(
                        self.actor_optim,
                        self.get_loss(states, actions),
                        self.actor.parameters(),
                    )
                }
            )
        return log_info


class DAggerDiscrete(DAggerContinuous):
    """Dataset Aggregation (DAgger) for Discrete Control
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_component(self):
        # load expert
        self.load_expert()

        # buffer
        buffer_kwarg = {
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "action_dtype": self.action_dtype,
            "device": self.device,
            "buffer_size": self.algo_config["buffer_size"],
        }
        self.trans_buffer = DAggerBuffer(**buffer_kwarg)

        # actor
        actor_kwarg = {
            "state_shape": self.state_shape,
            "net_arch": self.algo_config["actor"]["net_arch"],
            "action_shape": self.action_shape,
            "activation_fn": getattr(nn, self.algo_config["actor"]["activation_fn"]),
        }
        self.actor = MLPDeterministicActor(**actor_kwarg).to(self.device)
        self.actor_optim = getattr(optim, self.algo_config["actor"]["optimizer"])(
            self.actor.parameters(), self.algo_config["actor"]["lr"]
        )

        self.models.update({"actor": self.actor, "actor_optim": self.actor_optim})

    def get_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg,
    ):
        state = th.Tensor(state).to(self.device) if type(state) is np.ndarray else state
        prob = F.softmax(self.actor(state), dim=-1)  # unnormalized

        # We do not keep gradient.
        # To support back propagation, please consider using Gumbel-Softmax tricks
        if deterministic:
            action = th.argmax(prob, dim=-1)
        else:
            dist = Categorical(prob)
            action = dist.sample()  # sample

        if return_log_prob:
            log_prob = th.log(prob.gather(0, action))
        else:
            log_prob = None

        if not keep_dtype_tensor:
            action, log_prob = tensor2ndarray((action, log_prob))

        return (action, log_prob) if return_log_prob else action

    def get_loss(self, expert_states: th.Tensor, expert_actions: th.Tensor):
        predicted_actions = self.actor(expert_states)
        loss = F.cross_entropy(predicted_actions, expert_actions.squeeze())
        return loss