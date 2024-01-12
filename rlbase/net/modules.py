from typing import List, Tuple

import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn


def variable(shape: Tuple[int, ...]):
    return nn.Parameter(th.zeros(shape), requires_grad=True)


def mlp(
    input_shape: Tuple[int,],
    output_shape: Tuple[int,],
    net_arch: List[int],
    activation_fn: nn.Module = nn.ReLU,
    squash_output: bool = False,
) -> Tuple[List[nn.Module], int]:
    """
    :return: (net, feature_dim)
    """
    # output feature dimension
    if output_shape[0] == -1:
        if len(net_arch) > 0:
            feature_shape = (net_arch[-1], 0)
        else:
            raise ValueError("Empty MLP!")
    else:
        feature_shape = output_shape
    # networks
    net = nn.Sequential(
        *create_mlp(
            input_shape[0], output_shape[0], net_arch, activation_fn, squash_output
        )
    )
    return net, feature_shape


def cnn(
    input_shape: List[int],
    output_dim: int,
    net_arch: List[Tuple[int]],
    activation_fn: nn.Module = nn.ReLU,
) -> Tuple[List[nn.Module], int]:
    """
    :param input_shape: (channel, ...)
    :net_arch: list of conv2d, i.e., (output_channel, kernel_size, stride, padding) 
    """
    input_channel = input_shape[0]

    if len(net_arch) > 0:
        module = [nn.Conv2d(input_channel, *net_arch[0]), activation_fn()]
    else:
        raise ValueError("Empty CNN!")

    # parse modules
    for i in range(1, len(net_arch)):
        module.append(nn.Conv2d(net_arch[i - 1][0], *net_arch[i]))
        module.append(activation_fn())
    net = nn.Sequential(*module)
    net.add_module("flatten-0", nn.Flatten())

    # Compute shape by doing one forward pass
    with th.no_grad():
        n_flatten = net(th.randn(input_shape).unsqueeze(dim=0)).shape[1]

    # We use -1 to just extract the feature
    if output_dim == -1:
        return net, n_flatten
    else:
        net.add_module("linear-0", nn.Linear(n_flatten, output_dim))
        return net, output_dim
