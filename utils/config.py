import os
import torch
from yaml import load, FullLoader
import argparse

def load_yml(yml_file_path):
    if not os.path.exists(yml_file_path):  # check whether config file exists
        raise ValueError("Config file does not exists!")
    
    with open(yml_file_path, 'r', encoding='utf-8') as f:
        configs = load(f, Loader=FullLoader)
    return configs

def read_config(default_config='config/default.yml'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    configs = load_yml(args.config)
    configs['device'] =  torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if default_config is not None:  # preload defalut configs
        default_configs = load_yml(default_config)
        configs = {**default_configs, **configs}
    return configs