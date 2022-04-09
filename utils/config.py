import os
import torch
from yaml import load, dump, FullLoader
import argparse


def load_yml(yml_file_path):
    if not os.path.exists(yml_file_path):  # check whether config file exists
        raise ValueError("Config file does not exists!")

    with open(yml_file_path, "r", encoding="utf-8") as f:
        configs = load(f, Loader=FullLoader)
    return configs


def read_config(config_type="rl", config_dir="config"):
    # read config from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sac.yml")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # load yaml file
    custom_configs = load_yml(os.path.join(config_dir, args.config))
    custom_configs["device"] = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    common_configs = load_yml(os.path.join(config_dir, "_common.yml"))

    # merge all the configs
    configs = common_configs["share"]
    if config_type == "rl":
        configs = {**configs, **common_configs["rl"], **custom_configs}
    elif config_type == "il":
        configs = {**configs, **common_configs["il"], **custom_configs}
    else:
        raise ValueError("config_type should be 'rl' or 'il'")
    return configs


def write_config(configs, yml_file_path):
    with open(yml_file_path, "w", encoding="utf8") as f:
        dump(configs, f, allow_unicode=True)
