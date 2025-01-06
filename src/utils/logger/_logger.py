import json
import os
from datetime import datetime
from os.path import join
from typing import Any, Dict, List

import loguru
import tqdm
from tensorboardX import SummaryWriter

from ..ospy.file import copys


def _parse_record_param(
    args: Dict[str, Any], record_param: List[str]
) -> Dict[str, Any]:
    if args is None or record_param is None:
        return None
    else:
        record_param_dict = dict()
        for param in record_param:
            params = param.split(".")
            value = args
            for p in params:
                try:
                    value = value[p]
                except:
                    value = ""
                    break
            record_param_dict[param] = value
        return record_param_dict


def _get_exp_name(record_param_dict: Dict[str, Any], prefix: str = None):
    if prefix is not None:
        exp_name = prefix
    else:
        exp_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    for key, value in record_param_dict.items():
        if isinstance(value, str):
            value = "-".join(value.split(" "))
        exp_name = exp_name + f"~{key}={value}"
    return exp_name


class TBLogger:
    """Tensorboard Logger"""

    console = loguru.logger

    def __init__(
        self,
        work_dir: str = "./",
        args: Dict[str, Any] = {},
        root_log_dir: str = "runs",
        record_param: List[str] = [],
        backup_code: bool = False,
        code_files_list: List[str] = None,
        console_output: bool = True,
        **kwargs,
    ):
        """
        Args:
            work_dir: Path of the current work dir
            args: Hyper-parameters and configs
            root_log_dir: The root directory for all the logs
            record_param: Parameters used to name the log dir
            backup_code: Whether to backup code
            code_files_list: The list of code file/dir to backup
            console_output: Whether to output to the console
        """
        self.args = args
        self.record_param = record_param
        self.work_dir = os.path.abspath(work_dir)
        self.root_log_dir = join(work_dir, root_log_dir)
        self.code_files_list = code_files_list
        self.record_param_dict = _parse_record_param(args, record_param)
        self.tqdm = tqdm

        # create log dirs
        self.exp_name = _get_exp_name(self.record_param_dict)
        self.exp_dir = join(self.root_log_dir, self.exp_name)
        self._create_artifact_dir()

        # init tb
        self.tb = SummaryWriter(log_dir=self.exp_dir, **kwargs)

        # init loguru
        if not console_output:
            self.console.remove()
        self.console_log_file = join(self.exp_dir, "console.log")
        self.console.add(self.console_log_file, format="{time} -- {level} -- {message}")

        if backup_code:
            self._backup_code()

        # save arguments
        self._save_args()

    def _create_artifact_dir(self):
        self.ckpt_dir = join(self.exp_dir, "ckpt")
        os.makedirs(self.ckpt_dir)  # checkpoint, for model, data, etc.

        self.result_dir = join(self.exp_dir, "result")
        os.makedirs(self.result_dir)  # result, for some intermediate result

        self.code_bk_dir = join(self.exp_dir, "code")
        os.makedirs(self.code_bk_dir)  # back up code

    def _save_args(self):
        if self.args is None:
            return
        else:
            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(self.args)

            # self.console.info(f"Arguments: {self.args}")
            self.console.info(
                f"Save arguments to {join(self.exp_dir, 'parameter.json')}"
            )
            with open(join(self.exp_dir, "parameter.json"), "w") as f:
                jd = json.dumps(self.args, indent=4)
                print(jd, file=f)

    def _backup_code(self):
        for code in self.code_files_list:
            src_path = join(self.work_dir, code)
            tgt_path = join(self.code_bk_dir, code)
            copys(src_path, tgt_path)

    # ================ Additional Helper Functions ================

    def add_stats(self, stats: Dict[str, float], t: int):
        for key, value in stats.items():
            self.tb.add_scalar(key, value, t)
