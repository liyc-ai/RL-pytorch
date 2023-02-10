import os
from os.path import join

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.utils import set_random_seed

import ilkit
from ilkit.util.data import DataHandler
from ilkit.util.env import get_env_info, make_env, reset_env_fn
from ilkit.util.logger import setup_logger
from ilkit.util.ptu import clean_cuda, set_torch

# set global working dir
WORK_DIR = os.getcwd()


@hydra.main(
    config_path=join(WORK_DIR, "conf"),
    config_name="collect_demo",
    version_base="1.3.1",
)
def collect_demo(cfg: DictConfig):
    # prepare experiment
    cfg = OmegaConf.to_object(cfg)
    set_torch()
    clean_cuda()
    set_random_seed(cfg["seed"], True)
    cfg["work_dir"] = WORK_DIR

    # setup environment
    env = make_env(cfg["env"]["id"])
    cfg["env"].update(get_env_info(env))

    # setup logger
    logger = setup_logger(cfg)

    # create agent
    agent = ilkit.make(cfg, logger)
    agent.load_model()

    # instantiate data handler
    data_handler = DataHandler()

    # specify dataset name
    collect_info = cfg["collect"]
    save_name = collect_info["save_name"]
    if save_name == "":
        env_name = cfg["env"]["id"].lower()
        save_name = env_name + "_expert"

    # specify dataset dir
    save_dir = collect_info["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # collect demonstration
    ## Note: If n_traj and n_step are both specified, we will just consider the n_traj
    logger.dump2log("Start to collect demonstrations...")
    n_traj = collect_info["n_traj"]
    n_step = collect_info["n_step"]
    if n_traj > 0 and n_step > 0:
        n_step = 0
        logger.dump2log(
            "n_traj and n_step are both specified, but we will ignore n_step"
        )
    data_handler.collect_demo(
        agent, env, reset_env_fn, n_traj, n_step, save_dir, save_name
    )
    logger.dump2log(f"Succeed to save demonstrations at {join(save_dir, save_name)}!")


if __name__ == "__main__":
    # start to collect demonstrations
    collect_demo()
