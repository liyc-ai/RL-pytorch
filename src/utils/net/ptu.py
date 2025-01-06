from itertools import zip_longest
from os.path import join
from typing import Dict, Iterable, List, Tuple, Union

import torch as th
from torch import nn
from torch.optim import Optimizer

# --------------------- Setting --------------------


def set_torch(default_th_dtype: th.dtype = th.float32):
    th.set_default_dtype(default_th_dtype)
    th.utils.backcompat.broadcast_warning.enabled = True
    th.utils.backcompat.keepdim_warning.enabled = True
    th.set_float32_matmul_precision("high")


# --------------------- Tensor ---------------------


def tensor2ndarray(tensors: Tuple[th.Tensor]):
    """Convert torch.Tensor to numpy.ndarray"""
    result = []
    for item in tensors:
        if th.is_tensor(item):
            result.append(item.detach().cpu().numpy())
        else:
            result.append(item)
    return result


# ------------------- Manipulate NN Module ----------------------


def move_device(modules: List[th.nn.Module], device: Union[str, th.device]):
    """Move net to specified device"""
    for module in modules:
        module.to(device)


def freeze_net(nets: List[nn.Module]):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = False


def save_torch_model(
    models: Dict[str, Union[nn.Module, th.Tensor]],
    ckpt_dir: str,
    model_name: str = "models",
    file_ext: str = ".pt",
) -> str:
    """Save [Pytorch] model to a pre-specified path
    Note: Currently, only th.Tensor and th.nn.Module are supported.
    """
    model_name = model_name + file_ext
    model_path = join(ckpt_dir, model_name)
    state_dicts = {}
    for name, model in models.items():
        if isinstance(model, th.Tensor):
            state_dicts[name] = {name: model}
        else:
            state_dicts[name] = model.state_dict()
    th.save(state_dicts, model_path)
    return f"Successfully save model to {model_path}!"


def load_torch_model(
    models: Dict[str, Union[nn.Module, th.Tensor]], model_path: str
) -> str:
    """Load [Pytorch] model from a pre-specified path"""
    state_dicts = th.load(model_path, weights_only=True)
    for name, model in models.items():
        if isinstance(model, th.Tensor):
            models[name].data = state_dicts[name][name].data
        else:
            model.load_state_dict(state_dicts[name])
    return f"Successfully load model from {model_path}!"


def set_train_mode(models: Dict[str, Union[nn.Module, th.Tensor]]):
    """Set mode of the models to train"""
    for model in models:
        if isinstance(model, nn.Module):
            models[model].train()


def set_eval_mode(models: Dict[str, Union[nn.Module, th.Tensor]]):
    """Set mode of the models to eval"""
    for model in models:
        if isinstance(model, nn.Module):
            models[model].eval()


# copied from stable_baselines3
def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    params: Iterable[th.Tensor],
    target_params: Iterable[th.Tensor],
    tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


# ------------------ Initialization ----------------------------


def orthogonal_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# ----------------------- Optimization ----------------------------


def gradient_descent(
    net_optim: Optimizer,
    loss: th.Tensor,
    parameters: Union[th.Tensor, Iterable[th.Tensor]] = None,
    max_grad_norm: float = None,
    retain_graph: bool = False,
):
    """Update network parameters with gradient descent."""
    net_optim.zero_grad()
    loss.backward(retain_graph=retain_graph)

    # gradient clip
    if all([parameters, max_grad_norm]):
        th.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

    net_optim.step()
    return loss.item()


# ------------------------ Modules ------------------------


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: nn.Module = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Copied from stable_baselines

    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


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
