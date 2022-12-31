"""
A script to view data of experiments.
"""

import argparse

from RLA.easy_log.log_tools import ViewLogTool


def argsparser():
    parser = argparse.ArgumentParser("View Log")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--task_table_name", type=str)
    parser.add_argument("--regex", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argsparser()
    dlt = ViewLogTool(
        proj_root=args.data_root, task_table_name=args.task_table_name, regex=args.regex
    )
    dlt.view_log()
