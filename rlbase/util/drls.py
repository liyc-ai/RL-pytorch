import torch as th


class GAE:
    """Estimate Advantage using GAE (https://arxiv.org/abs/1506.02438)

    Ref:
    [1] https://nn.labml.ai/rl/ppo/gae.html
    [2] https://github.com/ikostrikov/pytorch-trpo
    """

    def __init__(
        self,
        gamma: float,
        lambda_: float,
        norm_adv: bool = True,
        use_td_lambda: bool = True,
    ):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.norm_adv = norm_adv
        self.use_td_lambda = use_td_lambda

    def __call__(
        self,
        value_net: th.nn.Module,
        states: th.Tensor,
        rewards: th.Tensor,
        next_states: th.Tensor,
        dones: th.Tensor,
    ):
        """Here we can use two different methods to calculate Returns"""
        not_dones = 1.0 - dones

        if self.use_td_lambda:
            Rs, advantages = self.td_lambda(
                value_net, states, rewards, next_states, not_dones
            )
        else:
            Rs, advantages = self.gae(
                value_net, states, rewards, next_states, not_dones
            )

        if self.norm_adv:
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return Rs, advantages

    def gae(
        self,
        value_net: th.nn.Module,
        states: th.Tensor,
        rewards: th.Tensor,
        next_states: th.Tensor,
        not_dones: th.Tensor,
    ):
        Rs = th.empty_like(rewards)  # reward-to-go R_t
        advantages = th.empty_like(rewards)  # advantage
        values = value_net(states)

        last_value = value_net(next_states[-1])
        last_return = th.clone(last_value)
        last_advantage = 0.0

        for t in reversed(range(rewards.shape[0])):
            # calculate rewards-to-go reward
            Rs[t] = rewards[t] + self.gamma * last_return * not_dones[t]
            # delta and advantage
            delta = rewards[t] + self.gamma * last_value * not_dones[t] - values[t]
            advantages[t] = (
                delta + self.gamma * self.lambda_ * not_dones[t] * last_advantage
            )
            # update pointer
            last_value = th.clone(values[t])
            last_advantage = advantages[t].clone()
            last_return = Rs[t].clone()
        return Rs, advantages

    def td_lambda(
        self,
        value_net: th.nn.Module,
        states: th.Tensor,
        rewards: th.Tensor,
        next_states: th.Tensor,
        not_dones: th.Tensor,
    ):
        # calcultae value
        values, next_values = value_net(states), value_net(next_states)
        # calculate TD errors.
        deltas = rewards + self.gamma * next_values * not_dones - values
        # initialize gae.
        advantages = th.empty_like(rewards)
        # calculate gae recursively from behind.
        advantages[-1] = deltas[-1]
        for t in reversed(range(rewards.size(0) - 1)):
            advantages[t] = (
                deltas[t] + self.gamma * self.lambda_ * not_dones[t] * advantages[t + 1]
            )

        return advantages + values, advantages
