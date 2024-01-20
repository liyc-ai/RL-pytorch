import os
from os.path import join

import hydra
from omegaconf import DictConfig, OmegaConf
from rlplugs.drls.env import get_env_info, make_env, reset_env_fn
from rlplugs.logger import TBLogger
from rlplugs.net.ptu import clean_cuda, save_torch_model, set_torch
from stable_baselines3.common.utils import set_random_seed

import rlpyt

# get global work dir
WORK_DIR = os.getcwd()


@hydra.main(
    config_path=join(WORK_DIR, "conf"), config_name="run_exp", version_base="1.3.1"
)
def main(cfg: DictConfig):
    cfg.work_dir = WORK_DIR
    # prepare experiment
    set_torch()
    clean_cuda()
    set_random_seed(cfg.seed, True)

    # setup logger
    logger = TBLogger(args=OmegaConf.to_object(cfg), record_param=cfg.log.record_param)

    # setup environment
    train_env, eval_env = (make_env(cfg.env.id), make_env(cfg.env.id))
    OmegaConf.update(cfg, "env[info]", get_env_info(eval_env), merge=False)

    # create agent
    agent = rlpyt.make(cfg, logger)
    # agent.load_model()

    # train agent
    agent.learn(train_env, eval_env, reset_env_fn)

    # save model
    logger.console.info(
        f"Successfully save models into {save_torch_model(agent.models, logger.ckpt_dir, 'final_model.pt')}"
    )


if __name__ == "__main__":
    main()
