import torch
import numpy as np
import random
from algo import ALGOS
from train_expert import train
from utils.config import load_yml_config

def set_random_seed(seed, env=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        
def get_expert(expert_model_path = '', expert_config='sac.yaml'):
    configs = load_yml_config(expert_config)
    if expert_model_path:
        expert = ALGOS[configs['algo']](configs)
        expert.load_model(expert_model_path)
    else:
        expert = train(configs)
    return expert
