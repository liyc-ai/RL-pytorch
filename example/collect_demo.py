import os
from os.path import join
from typing import Dict

import env_utils
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.utils import set_random_seed

import ilkit
from ilkit.util.data import DataHandler
from ilkit.util.osm import recursively_update
from ilkit.util.ptu import clean_cuda, set_torch

# set global working dir
ROOT_DIR = os.getcwd()

def prepare_exp(cfg: DictConfig) -> Dict:
    global ROOT_DIR
    set_torch()
    clean_cuda()
    set_random_seed(cfg["seed"], True)
    cfg["root_dir"] = ROOT_DIR
    return OmegaConf.to_object(cfg)

@hydra.main(config_path=join(ROOT_DIR, "conf"), config_name="_exp_config", version_base='1.3.1')
def collect_demo(cfg: DictConfig):
    # prepare experiment
    cfg = prepare_exp(cfg)
    
    # setup environment
    eval_env = env_utils.make_env(cfg["env"]["id"])
    env_info = env_utils.get_env_info(eval_env)
    env_info.update({ 
        "eval": eval_env,
        "reset_env": env_utils.reset_env,
    })
    cfg = recursively_update({"env": env_info}, cfg)
    
    # create agent
    agent = ilkit.make(cfg)
    agent.load_model(cfg["model_path"])
    
    # instantiate data handler
    data_handler = DataHandler()

    # specify dataset name
    collect_info = cfg["collect"]
    save_name = collect_info["save_name"]
    if save_name == "":
        env_name, version = cfg["env"]["id"].lower().split("-")
        save_name = env_name + "_expert-" + version

    # specify dataset dir
    save_dir = collect_info["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # collect demonstration
    ## Note: If n_traj and n_step are both specified, we will just consider the n_traj
    agent.logger.info("Start to collect demonstrations...")
    n_traj = collect_info["n_traj"]
    n_step = collect_info["n_step"]
    if n_traj > 0 and n_step > 0:
        n_step = 0
        agent.logger.warn(
            "n_traj and n_step are both specified, but we will ignore n_step"
        )
    data_handler.collect_demo(agent, n_traj, n_step, save_dir, save_name)
    agent.logger.info(
        f"Succeed to save demonstrations at {join(save_dir, save_name)}!"
    )


if __name__ == "__main__":
    # start to collect demonstrations
    collect_demo()
