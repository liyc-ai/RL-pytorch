import os
from os.path import join

from ..ospy.file import copys


def archive_logs(exp_name: str, src_dir: str, tgt_dir: str = "archived"):
    """Locally archive src_dir/exp_name to tgt_dir"""
    os.makedirs(tgt_dir, exist_ok=True)
    copys(join(src_dir, exp_name), join(tgt_dir, exp_name))
