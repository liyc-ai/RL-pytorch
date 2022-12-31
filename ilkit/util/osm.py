"""Os manipulator, namely, operations associated with path, python object, etc."""
import os
from typing import Dict

# --------------------- Path -------------------------


def back_path(path: str, level: int):
    """Back the given path

    :param path: usually be __file__ of some specific file
    :param level: backward [level] times, e.g., 
    given file /home/ubuntu/examples/example.py and level 1, return /home/ubuntu/examples/
    """
    for _ in range(level):
        path = os.path.split(path)[0]
    return path

# --------------------- Python Object -------------------------

def recursively_update(src_dict: Dict, tgt_dict: Dict):
    """Recursively update tgt_dict with src_dict
    """
    for key, value in src_dict.items():
        if key in tgt_dict:
            if type(value) == dict:
                if type(tgt_dict[key]) == dict:
                    tgt_dict[key] = recursively_update(value, tgt_dict[key])
                else:
                    tgt_dict[key] = value
            else:
                tgt_dict[key] = value
        else:
            tgt_dict[key] = value
    return tgt_dict