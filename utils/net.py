import torch.nn as nn

def build_mlp_extractor(input_dim, hidden_size, activation_fn):
    """
    Create MLP feature extractor, code modified from:
    
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
    """
    if len(hidden_size) > 0:
        mlp_extractor = [nn.Linear(input_dim, hidden_size[0]), activation_fn()]
    else:
        mlp_extractor = []

    for idx in range(len(hidden_size) - 1):
        mlp_extractor.append(nn.Linear(hidden_size[idx], hidden_size[idx + 1]))
        mlp_extractor.append(activation_fn())
        
    return mlp_extractor