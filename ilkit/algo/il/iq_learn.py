from typing import Dict, Tuple, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

from ilkit.algo.rl.dqn import DQN
from ilkit.algo.rl.sac import SAC
from ilkit.util.logger import BaseLogger
from ilkit.util.ptu import tensor2ndarray


# phi_fn implementations
def fkl(x: th.Tensor) -> th.Tensor:
    """Forward KL
    """
    return 1 + th.log(x)


def rkl(x: th.Tensor) -> th.Tensor:
    """Reverse KL
    """
    return -th.exp(-x - 1)


def hellinger(x: th.Tensor) -> th.Tensor:
    """Squared Hellinger
    """
    return x / (1 + x)


def chi(x: th.Tensor) -> th.Tensor:
    """Pearson chi^2
    """
    return x - th.pow(x, 2) / 4


def tv(x: th.Tensor) -> th.Tensor:
    """Total Variation
    """
    return x


def js(x: th.Tensor) -> th.Tensor:
    """Jenson-Shannon
    """
    return th.log(2 - th.exp(-x))


class IQLearnDiscrete(DQN):
    """Inverse soft-Q Learning for Imitation (IQ-Learn), discrete action space
    """

    def __init__(self, cfg: Dict, logger: BaseLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        super().setup_model()

        # hyper-param
        self.alpha = self.algo_cfg["alpha"]

    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        **kwarg
    ) -> Union[th.Tensor, th.Tensor]:

        # softq's optimal policy
        with th.no_grad():
            state = (
                th.Tensor(state).to(self.device) if type(state) is np.ndarray else state
            )
            q = self.q_net(state)
            dist = F.softmax(q / self.alpha, dim=-1)

        # sample action
        if deterministic:
            dist = Categorical(dist)
            action = dist.sample()
        else:
            action = th.argmax(dist, dim=-1)

        # tensor or ndarray
        if not keep_dtype_tensor:
            action, = tensor2ndarray((action,))

        return action

    def _get_v(self, state: th.Tensor) -> th.Tensor:
        """Soft V
        """
        q = self.q_net(state)
        v = self.alpha * th.logsumexp(q / self.alpha, dim=-11, keepdim=True)
        return v

    def update(self) -> Dict:
        self.log_info = dict()
        if self.trans_buffer.size >= self.batch_size:
            states, actions, next_states, rewards, dones = self.trans_buffer.sample(
                self.batch_size, shuffle=True
            )

            self.log_info.update({})

        return self.log_info


class IQLearnContinuous(SAC):
    """Inverse soft-Q Learning for Imitation (IQ-Learn), continuous action space
    """

    def __init__(self, cfg: Dict, logger: BaseLogger):
        super().__init__(cfg, logger)

    def setup_model(self):
        ...

    def select_action(
        self,
        state: Union[np.ndarray, th.Tensor],
        deterministic: bool,
        keep_dtype_tensor: bool,
        return_log_prob: bool,
        **kwarg
    ) -> Union[np.ndarray, th.Tensor]:
        ...

    def update(self):
        ...
