"""
A script to archive benchmarking experiments. 

It is convenient to merge the archived experiments and the current task into tensorboard by:

tensorboard --logdir ./log/your_task/,./log/archived/

"""

import argparse

from RLA.easy_log.log_tools import ArchiveLogTool


def argsparser():
    parser = argparse.ArgumentParser("Archive Log")
    # reduce setting
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--task_table_name", type=str)
    parser.add_argument("--regex", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    dlt = ArchiveLogTool(
        proj_root=args.data_root, task_table_name=args.task_table_name, regex=args.regex
    )
    dlt.archive_log()
