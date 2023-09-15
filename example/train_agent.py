import os
from os.path import join

import hydra
from mllogger import TBLogger
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.utils import set_random_seed

import ilkit
from ilkit.util.env import get_env_info, make_env, reset_env_fn
from ilkit.util.ptu import clean_cuda, set_torch

# get global work dir
WORK_DIR = os.getcwd()


@hydra.main(
    config_path=join(WORK_DIR, "conf"), config_name="run_exp", version_base="1.3.1"
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

    # setup logger
    logger = TBLogger(
        args=cfg, record_param=cfg["log"]["record_param"]
    )

    # create agent
    agent = ilkit.make(cfg, logger)
    # agent.load_model()

    # train agent
    agent.learn(train_env, eval_env, reset_env_fn)

    # save model
    agent.save_model(join(getattr(logger, "checkpoint_dir"), "final_model.pt"))


if __name__ == "__main__":
    main()
