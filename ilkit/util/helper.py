"""Helper functions associated with sys, python object, etc."""
import shutil
from os.path import isfile, join, split
from typing import Dict

import hydra
from omegaconf import OmegaConf

# --------------------- Sys -------------------------


def back_path(path: str, level: int):
    """Back the given path

    :param path: usually be __file__ of some specific file
    :param level: backward [level] times, e.g., 
    given file /home/ubuntu/examples/example.py and level 1, return /home/ubuntu/examples/
    """
    for _ in range(level):
        path = split(path)[0]
    return path


def copy_file_dir(src_dir: str, dst_dir: str, file: str):
    src_file_path = join(src_dir, *file.split("/"))
    if isfile(src_file_path):
        shutil.copy(src_file_path, dst_dir)
    else:
        shutil.copytree(src_file_path, dst_dir)


# --------------------- Python Helpers -------------------------


def re_update_dict(src_dict: Dict, tgt_dict: Dict):
    """Recursively update tgt_dict with src_dict
    """
    for key, value in src_dict.items():
        if key in tgt_dict:
            if type(value) == dict:
                if type(tgt_dict[key]) == dict:
                    tgt_dict[key] = re_update_dict(value, tgt_dict[key])
                else:
                    tgt_dict[key] = value
            else:
                tgt_dict[key] = value
        else:
            tgt_dict[key] = value
    return tgt_dict


# --------------------- Hydra Helpers ----------------------------


def load_cfg(work_dir: str):
    c = []
    hydra.main(
        config_path=join(work_dir, "conf"),
        config_name="_exp_config.yaml",
        version_base="1.3.1",
    )(lambda x: c.append(x))()
    cfg = c[0]
    cfg.work_dir = work_dir
    cfg = OmegaConf.to_object(cfg)
    return cfg
