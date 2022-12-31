import time
from os.path import join
from typing import Dict, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from ilkit.algo.il import ILPolicy
from ilkit.net.actor import MLPDeterministicActor, MLPGaussianActor
from ilkit.util.eval import eval_policy
from ilkit.util.ptu import gradient_descent, tensor2ndarray


class BCContinuous(ILPolicy):
    """Behavioral Cloning (BC) for Continuous Control
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_param(self):
        self.batch_size = self.algo_config["batch_size"]

    def init_component(self):
        # load expert dataset()
        self.load_expert()

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

    def get_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg,
    ):
        return self.actor.sample(
            state, deterministic, keep_dtype_tensor, return_log_prob, self.device
        )

    def learn(self):
        if not self.cfg["train"]["learn"]:
            self.logger.warn("We did not learn anything!")
            return

        best_return = -float("inf")
        past_time = 0
        now_time = time.time()
        train_steps = self.cfg["train"]["max_steps"]
        eval_interval = self.cfg["train"]["eval_interval"]

        for t in range(train_steps):
            last_time = now_time
            self.exp_manager.time_step_holder.set_time(t)

            # update actor
            info = self.update()
            self.logger.logkvs(info)

            # evaluate
            if (t + 1) % eval_interval == 0:
                eval_return = eval_policy(self.eval_env, self.reset_env, self, self.seed)
                self.logger.logkv("eval/return", eval_return)

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
        states, actions = self.expert_buffer.sample(self.batch_size)[:2]
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

    def get_loss(self, expert_states: th.Tensor, expert_actions: th.Tensor):
        action_mean, action_std = self.actor(expert_states)
        log_prob = th.sum(
            Normal(action_mean, action_std).log_prob(expert_actions),
            dim=-1,
            keepdim=True,
        )
        loss = -th.mean(log_prob)

        # # orthogonal regularization
        # from imitation_base.util.ptu import orthogonal_reg
        # loss += orthogonal_reg(self.actor, self.device)

        return loss


class BCDiscrete(BCContinuous):
    """Behavioral Cloning (BC) for Discrete Control
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_component(self):
        # load expert dataset
        self.load_expert()

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
