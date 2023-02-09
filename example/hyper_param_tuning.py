import os
from os.path import join
from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.utils import set_random_seed

import ilkit
from ilkit.util.env import get_env_info, make_env, reset_env
from ilkit.util.ptu import clean_cuda, set_torch

# get global working dir
WORK_DIR = os.getcwd()


def nni_optim_fn(cfg: Dict):
    """For nni, DQN example
    """
    import nni

    optimized_param = nni.get_next_parameter()
    cfg["agent"]["buffer_size"] = optimized_param["buffer_size"]
    cfg["agent"]["gamma"] = optimized_param["gamma"]
    cfg["agent"]["QNet"]["lr"] = optimized_param["lr"]


@hydra.main(
    config_path=join(WORK_DIR, "conf"), config_name="_run_exp", version_base="1.3.1"
)
def main(cfg: DictConfig):
    # prepare experiment
    cfg = OmegaConf.to_object(cfg)
    set_torch()
    clean_cuda()
    set_random_seed(cfg["seed"], True)
    cfg["work_dir"] = WORK_DIR

    # setup environment
    train_env, eval_env = (make_env(cfg["env"]["id"]), make_env(cfg["env"]["id"]))
    cfg["env"].update(get_env_info(eval_env))

    # hyper-param optimization
    nni_optim_fn(cfg)

    # create agent
    agent = ilkit.make(cfg)

    # train agent
    agent.learn(train_env, eval_env, reset_env)


if __name__ == "__main__":
    main()
