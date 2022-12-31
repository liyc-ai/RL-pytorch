import os
from os.path import join
from typing import Dict

import env_utils
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.utils import set_random_seed

import ilkit
from ilkit.util.osm import recursively_update
from ilkit.util.ptu import clean_cuda, set_torch

# set global working dir
ROOT_DIR = os.getcwd()

def nni_optim_fn(cfg: Dict):
    """For nni, DQN example
    """
    import nni
    optimized_param = nni.get_next_parameter()
    cfg["agent"]["buffer_size"] = optimized_param["buffer_size"]
    cfg["agent"]["gamma"] = optimized_param["gamma"]
    cfg["agent"]["QNet"]["lr"] = optimized_param["lr"]

def prepare_exp(cfg: DictConfig) -> Dict:
    global ROOT_DIR
    set_torch()
    clean_cuda()
    set_random_seed(cfg["seed"], True)
    cfg["root_dir"] = ROOT_DIR
    return OmegaConf.to_object(cfg)

@hydra.main(config_path=join(ROOT_DIR, "conf"), config_name="_exp_config", version_base='1.3.1')
def main(cfg: DictConfig):
    # prepare experiment
    cfg = prepare_exp(cfg)
    
    # setup environment
    train_env, eval_env = env_utils.make_env(cfg["env"]["id"]), env_utils.make_env(cfg["env"]["id"])
    env_info = env_utils.get_env_info(eval_env)
    env_info.update({
        "train": train_env, 
        "eval": eval_env,
        "reset_env": env_utils.reset_env,
    })
    cfg = recursively_update({"env": env_info}, cfg)
    
    # hyper-param optimization
    if cfg["hpo"]: nni_optim_fn(cfg)
    
    # create agent
    agent = ilkit.make(cfg)
    agent.load_model(cfg["model_path"])
    
    # setup experiment manager
    agent.setup_rla()

    # train agent
    agent.learn()

    # save model
    agent.save_model(join(agent.checkpoint_dir, "final_model.pt"))

if __name__ == "__main__":
    main()