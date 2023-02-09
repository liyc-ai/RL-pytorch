import shutil
from os.path import join

from hydra import compose, initialize
from omegaconf import OmegaConf

from ilkit.util.helper import back_path
from ilkit.util.logger import setup_logger

LOGS = "logs_test"


def test_logger() -> None:
    with initialize(version_base="1.3.1", config_path="../conf"):
        cfg = compose(config_name="run_exp")
        cfg = OmegaConf.to_object(cfg)
        work_dir = back_path(__file__, 2)
        cfg["work_dir"] = work_dir
    supported_logger = ["rla", "tensorboard", "wandb"]
    for type_ in supported_logger:
        cfg["log"]["logger_type"] = type_
        try:
            setup_logger(cfg, LOGS)
        except Exception as exc:
            assert False, f"Create logger with type {type_} raised an exception {exc}."
        finally:
            shutil.rmtree(join(work_dir, LOGS))
