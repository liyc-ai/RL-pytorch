from typing import List, Tuple

import torch as th
from torch.nn import Module, ReLU

from ilkit.net.modules import mlp
from ilkit.util.ptu import orthogonal_init_


class MLPCritic(Module):
    def __init__(
        self,
        input_shape: Tuple[int,],
        output_shape: Tuple[int,],
        net_arch: List[int],
        activation_fn: Module = ReLU,
        **kwarg
    ):
        """
        :param input_dim: input dimension (for vector) or input channel (for image)
        """
        super().__init__()
        self.value_net, _ = mlp(
            input_shape, output_shape, net_arch, activation_fn, **kwarg
        )
        self.apply(orthogonal_init_)

    def forward(self, *input_):
        input_ = th.cat(input_, dim=-1)
        return self.value_net(input_)


class MLPTwinCritic(Module):
    def __init__(
        self,
        input_shape: Tuple[int,],
        output_shape: Tuple[int,],
        net_arch: List[int],
        activation_fn: Module = ReLU,
        **kwarg
    ):
        super().__init__()

        self.Q_1, _ = mlp(input_shape, output_shape, net_arch, activation_fn, **kwarg)
        self.Q_2, _ = mlp(input_shape, output_shape, net_arch, activation_fn, **kwarg)

        self.apply(orthogonal_init_)

    def forward(self, twin_value: bool, *input_):
        """
        :param twin_value: whether to return both Q1 and Q2
        """
        input_ = th.cat(input_, dim=-1)
        return (self.Q_1(input_), self.Q_2(input_)) if twin_value else self.Q_1(input_)


class MLPDuleQNet(Module):
    """Dueling Q Network
    """

    def __init__(
        self,
        input_shape: Tuple[int,],
        output_shape: Tuple[int,],
        net_arch: List[int],
        v_head: List[int],
        adv_head: List[int],
        activation_fn: Module = ReLU,
        mix_type: str = "max",
        **kwarg
    ):
        super().__init__()
        self.feature_extrator, feature_shape = mlp(
            input_shape, (-1,), net_arch, activation_fn, **kwarg
        )
        self.value_head, _ = mlp(feature_shape, (1,), v_head, activation_fn, **kwarg)
        self.adv_head, _ = mlp(
            feature_shape, output_shape, adv_head, activation_fn, **kwarg
        )
        self.mix_type = mix_type

        self.apply(orthogonal_init_)

    def forward(self, state: th.Tensor):
        feature = self.feature_extrator(state)
        v = self.value_head(feature)
        adv = self.adv_head(feature)
        if self.mix_type == "max":
            q = v + (adv - th.max(adv, dim=-1, keepdim=True)[0])
        elif self.mix_type == "mean":
            q = v + (adv - th.mean(adv, dim=-1, keepdim=True))
        else:
            raise NotImplementedError
        return q
