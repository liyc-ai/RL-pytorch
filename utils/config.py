import os
import sys
from yaml import load, FullLoader

config_dir = './config'

#! argparse style for parameter passing
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type='string', default='default.yaml', 
#                     help='Config file name, with file extension')
# args = parser.parse_args()
# config_file_name = args.config

def read_config():
    config_file_name = sys.argv[1] + '.yml'
        
    config_file_path = os.path.join(config_dir, config_file_name)
    if not os.path.exists(config_file_path):  # check whether config file exists
        raise ValueError("Config file does not exists!")
    
    with open(config_file_path, 'r', encoding='utf-8') as f:
        configs = load(f, Loader=FullLoader)
    return configs