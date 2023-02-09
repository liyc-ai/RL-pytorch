import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from os.path import join
from typing import Any, Dict

from omegaconf import OmegaConf

from ilkit.util.helper import copy_file_dir

try:
    import RLA
    import wandb
    from tensorboardX import SummaryWriter
except:
    pass


class BaseLogger(ABC):
    def __init__(self, cfg: Dict[str, Any], log_root: str):
        """
        : param root: Path of all logs
        : param record_params: Used to name current results
        """
        super().__init__()
        self.cfg = cfg
        self.log_cfg = self.cfg["log"]
        self.work_dir = cfg["work_dir"]
        self.checkpoint_dir = None  #! Must be specified later
        self.log_root = log_root
        self.create_logger()

    @abstractmethod
    def create_logger(self):
        raise NotImplementedError

    @abstractmethod
    def set_global_t(self, t: int):
        raise NotImplementedError

    @abstractmethod
    def get_global_t(self):
        raise NotImplementedError

    @abstractmethod
    def dump2log(self, info: Any):
        """Log [info] to log file
        """
        raise NotImplementedError

    @abstractmethod
    def logkv(self, key: str, value: Any, tag: str = None):
        raise NotImplementedError

    @abstractmethod
    def logkvs(self, infos: Dict[str, Any], tag: str = None):
        raise NotImplementedError

    @abstractmethod
    def dumpkvs(self):
        raise NotImplementedError


class TBLogger(BaseLogger):
    def __init__(self, cfg: Dict[str, Any], log_root: str):
        super().__init__(cfg, log_root)
        self.global_t = 0
        self.kvs = dict()

    def create_logger(self):
        self._create_log_dir()

        # tensorboard logger
        self._create_tb_logger()

        # file logger
        self._create_root_logger()

        # checkpoint dir
        self._create_ckpt()

        # backup
        self._backup()

    def set_global_t(self, t: int):
        self.global_t = t

    def get_global_t(self):
        return self.global_t

    def logkv(self, key: str, value: Any, tag: str = None):
        if tag is not None:
            self.kvs[f"{tag}/{key}"] = value
        else:
            self.kvs[key] = value

    def logkvs(self, infos: Dict[str, Any], tag: str = None):
        for key, value in infos.items():
            self.logkv(key, value, tag)

    def dumpkvs(self):
        for key, value in self.kvs.items():
            self.tb_logger.add_scalar(key, value, self.global_t)
        self.kvs = dict()

    def dump2log(self, info: str):
        getattr(self.root_logger, "info")(info)

    def _parse_record_param(self):
        self._record_param: Dict[str, Any] = dict()
        if self.cfg["log"]["record_param"] is not None:
            for key in self.log_cfg["record_param"]:
                keys = key.split(".")
                _value = self.cfg[keys[0]]
                for _key in keys[1:]:
                    _value = _value[_key]
                self._record_param[key] = _value

    def _create_log_dir(self):
        self._parse_record_param()

        self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for key in sorted(self._record_param.keys()):
            self.run_name = (
                self.run_name + "&" + key + "=" + str(self._record_param[key])
            )
        self.log_dir = join(
            self.work_dir, self.log_root, self.log_cfg["project"], self.run_name
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def _create_tb_logger(self):
        self.tb_logger = SummaryWriter(self.log_dir)

    def _create_root_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(join(self.log_dir, "log_file.log")),
                logging.StreamHandler(),
            ],
        )
        self.root_logger = logging.getLogger(__name__)

    def _create_ckpt(self):
        self.checkpoint_dir = join(self.log_dir, "checkpoint")  # for model
        self.results_dir = join(self.log_dir, "results")  # for intermediate data
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def _backup(self):
        # parameters
        with open(join(self.log_dir, "parameter.json"), "w") as f:
            jd = json.dumps(self.cfg, indent=4)
            print(jd, file=f)

        # code
        dst_code_dir = join(self.log_dir, "code")
        os.makedirs(dst_code_dir, exist_ok=True)
        if self.log_cfg["backup_code_dir"] is not None:  # code dir
            for code_dir in self.log_cfg["backup_code_dir"]:
                copy_file_dir(self.work_dir, dst_code_dir, code_dir)
        if self.log_cfg["run_file"] is not None:  # code file
            if type(self.log_cfg["run_file"]) is list:
                for file in self.log_cfg["run_file"]:
                    copy_file_dir(self.work_dir, dst_code_dir, file)
            else:
                copy_file_dir(self.work_dir, dst_code_dir, self.log_cfg["run_file"])


class RLALogger(BaseLogger):
    def __init__(self, cfg: Dict[str, Any], log_root: str):
        super().__init__(cfg, log_root)

    def create_logger(self):
        self.logger = RLA.logger
        self.exp_manager = RLA.exp_manager

        # exp_manager.new_saver()
        # exp_manager.save_checkpoint()

        rla_config_path = join(self.work_dir, *self.log_cfg["rla_config"].split("/"))
        self.exp_manager.configure(
            task_table_name=self.log_cfg["project"],
            private_config_path=rla_config_path,
            data_root=join(self.work_dir, self.log_root),
            run_file=self.log_cfg["run_file"],
            code_root=self.work_dir,
        )
        self.exp_manager.set_hyper_param(**self.cfg)
        self.exp_manager.add_record_param(self.log_cfg["record_param"])
        self.exp_manager.log_files_gen()
        self.exp_manager.print_args()

        self.results_dir = self.exp_manager.results_dir
        self.checkpoint_dir = self.exp_manager.checkpoint_dir

    def set_global_t(self, t: int):
        self.exp_manager.time_step_holder.set_time(t)

    def get_global_t(self):
        return self.exp_manager.time_step_holder.get_time()

    def dump2log(self, info: Any):
        """Log [info] to log file
        """
        self.logger.info(info)

    def logkv(self, key: str, value: Any, tag: str = None):
        if tag is not None:
            self.logger.logkv(f"{tag}/{key}", value)
        else:
            self.logger.logkv(key, value)

    def logkvs(self, infos: Dict[str, Any], tag: str = None):
        self.logger.logkvs(infos, f"{tag}/")

    def dumpkvs(self):
        self.logger.dumpkvs()


class WBLogger(TBLogger):
    def __init__(self, cfg: Dict[str, Any], log_root: str):
        super().__init__(cfg, log_root)

    def create_logger(self):
        # login with api keys
        os.environ["WANDB_API_KEY"] = OmegaConf.load(
            join(self.work_dir, *self.log_cfg["api_key_file"].split("/"))
        )["api_key"]

        # setup sync mode
        if self.log_cfg["up2cloud"]:
            os.environ["WANDB_MODE"] = "online"
        else:
            os.environ["WANDB_MODE"] = "offline"

        # other configs
        os.environ["WANDB_DISABLE_GIT"] = "true"
        # os.environ["WANDB_DISABLE_CODE"] = "true"

        # log dir
        self._create_log_dir()

        # file logger
        self._create_root_logger()

        # checkpoint dir
        self._create_ckpt()

        # backup code
        self._backup()

        # init
        wandb_cfg = dict(
            project=self.log_cfg["project"],
            name=self.run_name,
            dir=self.log_dir,
            reinit=True,
            config=self.cfg,
        )
        if self.log_cfg["entity"] is not None:
            wandb_cfg.update({"entity": self.log_cfg["entity"]})
        wandb.init(**wandb_cfg)

    def logkv(self, key: str, value: Any, tag: str = None):
        if tag is not None:
            wandb.log({f"{tag}/{key}": value}, step=self.global_t, commit=False)
        else:
            wandb.log({key: value}, step=self.global_t, commit=False)

    def logkvs(self, infos: Dict[str, Any], tag: str = None):
        if tag is not None:
            infos = {f"{tag}/{key}": value for (key, value) in infos.items()}
        wandb.log(infos, step=self.global_t, commit=False)

    def dumpkvs(self):
        wandb.log({"timestep": self.global_t}, commit=True)


def setup_logger(cfg: Dict, log_root: str = "logs") -> BaseLogger:
    if not cfg["log"]["setup_logger"]:
        return None
    # sanity check
    supported_logger = {"tensorboard": TBLogger, "rla": RLALogger, "wandb": WBLogger}
    logger_type = cfg["log"]["logger_type"]
    assert logger_type in supported_logger, f"Unsupported logger type {logger_type}"

    # instantiate the specified logger
    return supported_logger[logger_type](cfg, log_root)
