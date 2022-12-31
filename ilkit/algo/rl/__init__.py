import time
from os.path import join
from typing import Dict

from ilkit.algo import BasePolicy
from ilkit.util.data import TransitionBuffer
from ilkit.util.eval import eval_policy


class OnlineRLPolicy(BasePolicy):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # hyper-param
        self.batch_size = self.algo_config["batch_size"]
        self.gamma = self.algo_config["gamma"]

        # buffer
        buffer_kwarg = {
            "state_shape": self.state_shape,
            "action_shape": self.action_shape,
            "action_dtype": self.action_dtype,
            "device": self.device,
            "buffer_size": self.algo_config["buffer_size"],
        }
        self.trans_buffer = TransitionBuffer(**buffer_kwarg)

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
            if "warmup_steps" in self.algo_config and t < self.algo_config["warmup_steps"]:
                action = self.train_env.action_space.sample()
            else:
                action = self.get_action(
                    state,
                    keep_dtype_tensor=False,
                    deterministic=False,
                    return_log_prob=False,
                )
            next_state, reward, terminated, truncated, _ = self.train_env.step(action)
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
                next_state, info = self.reset_env(self.train_env, self.seed)
                train_return = 0

            # evaluate
            if (t + 1) % eval_interval == 0:
                eval_return = eval_policy(self.eval_env, self.reset_env, self, self.seed)
                self.logger.logkv("return/eval", eval_return)

                # hpo
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
