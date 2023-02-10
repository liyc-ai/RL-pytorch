import shutil
from os.path import exists, join

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from ilkit import IL_AGENTS, RL_AGENTS, make
from ilkit.util.env import get_env_info, make_env
from ilkit.util.logger import setup_logger
import os

LOGS = "logs_test"


def test_make_agent():
    work_dir = os.getcwd()
    logs_dir = join(work_dir, LOGS)

    # setup environment
    discrete_env = make_env("CartPole-v1")
    continuous_env = make_env("Hopper-v4")

    discrete_agents = ["dqn", "ddqn", "dueldqn", "bc_discrete", "dagger_discrete"]

    def _helper(agent: str, cfg: DictConfig):
        try:
            cfg = OmegaConf.to_object(cfg)
            cfg["work_dir"] = work_dir
            cfg["log"]["root"] = LOGS

            if agent in discrete_agents:
                cfg["env"].update(get_env_info(discrete_env))
                cfg["expert_dataset"][
                    "dataset_file_path"
                ] = "data/cartpole-v1_expert.hdf5"
            else:
                cfg["env"].update(get_env_info(continuous_env))
                cfg["expert_dataset"]["d4rl_env_id"] = "hopper-expert-v2"

            logger = setup_logger(cfg)
            make(cfg, logger)
        except Exception as exc:
            if exists(logs_dir):
                shutil.rmtree(logs_dir)
            assert False, f"Making {key} agent raised an exception {exc}."

    with initialize(version_base="1.3.1", config_path="../conf"):
        for key in RL_AGENTS:
            cfg = compose(config_name="run_exp", overrides=[f"agent=rl/{key}"])
            _helper(key, cfg)
        for key in IL_AGENTS:
            cfg = compose(config_name="run_exp", overrides=[f"agent=il/{key}"])
            _helper(key, cfg)

    if exists(logs_dir):
        shutil.rmtree(join(work_dir, LOGS))
