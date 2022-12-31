from typing import List, Tuple, Union

import numpy as np
import torch as th
from torch.distributions.normal import Normal
from torch.nn import Module, ReLU

from ilkit.net.modules import mlp, variable
from ilkit.util.ptu import tensor2ndarray

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class MLPGaussianActor(Module):
    """
    Gaussian actor for continuous action space
    """

    def __init__(
        self,
        state_shape: Tuple[int,...],
        action_shape: Tuple[int,...],
        net_arch: List[int],
        state_std_independent: bool = False,
        activation_fn: Module = ReLU,
        **kwarg
    ):
        """ 
        :param state_std_independent: whether std is a function of state
        """
        super().__init__()

        # network definition
        self.feature_extractor, feature_shape = mlp(
            state_shape, (-1,), net_arch, activation_fn, **kwarg
        )
        self.mu, _ = mlp(
            feature_shape, action_shape, [], activation_fn, **kwarg
        )

        # to unify self.log_std to be a function
        if state_std_independent:
            self._log_std = variable((1,) + action_shape)
            self.log_std = lambda _: self._log_std
        else:
            self.log_std, _ = mlp(
            feature_shape, action_shape, [], activation_fn, **kwarg
        )

    def forward(self, state: th.Tensor):
        feature = self.feature_extractor(state)
        mu, log_std = self.mu(feature), self.log_std(feature)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std.exp()

    def sample(
        self,
        state: Union[th.Tensor, np.ndarray],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        device: Union[th.device, str],
    ):
        state = th.Tensor(state).to(device) if type(state) is np.ndarray else state

        action_mean, action_std = self.forward(state)
        dist = Normal(action_mean, action_std)

        if deterministic:
            x = action_mean
        else:
            x = dist.rsample()

        if return_log_prob:
            log_prob = th.sum(dist.log_prob(x), axis=-1, keepdims=True)
        else:
            log_prob = None

        if not keep_dtype_tensor:
            x, log_prob = tensor2ndarray((x, log_prob))

        return (x, log_prob) if return_log_prob else x


class MLPDeterministicActor(Module):
    def __init__(
        self,
        state_shape: Tuple[int,],
        action_shape: Tuple[int,], 
        net_arch: List[int],
        activation_fn: Module = ReLU,
        **kwarg
    ):
        super().__init__()

        self.feature_extractor, feature_shape = mlp(
            state_shape, (-1,), net_arch, activation_fn, **kwarg
        )
        self.output_head, _ = mlp(
            feature_shape, action_shape, [], activation_fn, **kwarg
        )

    def forward(self, state: th.Tensor):
        feature = self.feature_extractor(state)
        action = self.output_head(feature)
        return action
