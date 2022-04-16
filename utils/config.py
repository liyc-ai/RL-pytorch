import os
from yaml import load, dump, FullLoader
import argparse


def load_yml_config(yml_file_name, config_dir="config"):
    yml_file_path = os.path.join(config_dir, yml_file_name)
    if not os.path.exists(yml_file_path):  # check whether config file exists
        raise ValueError("Config file does not exists!")

    with open(yml_file_path, "r", encoding="utf-8") as f:
        configs = load(f, Loader=FullLoader)
    return configs


def parse_args():
    # read config from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sac.yml")
    parser.add_argument("--generate-data", dest="g", action="store_true", default=False)
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v3")
    args = parser.parse_args()
    return args


def write_config(configs, yml_file_path):
    with open(yml_file_path, "w", encoding="utf8") as f:
        dump(configs, f, allow_unicode=True)
