import os
import shutil


def copys(src_path: str, tgt_path: str):
    if os.path.isfile(src_path):
        shutil.copy(src_path, tgt_path)
    elif os.path.isdir(src_path):
        shutil.copytree(src_path, tgt_path)
    else:
        raise TypeError("Unknown code file type!")
