import copy
import itertools
import torch
import torch.nn.functional as F
from algo.rl.ddpg import DDPGAgent
from net.actor import DeterministicActor
from net.critic import Critic
from utils.net import soft_update


class TD3Agent(DDPGAgent):
    """Twin Delayed Deep Deterministic Policy Gradient

    Code modified from: https://github.com/sfujim/TD3
    """

    def __init__(self, configs: dict):
        super().__init__(configs)
        self.policy_delay = configs["policy_delay"]
        # noise injection
        self.c = configs.get("c") * self.action_high
        self.sigma = configs.get("sigma") * self.action_high
        self.total_it = 0
        # actor
        self.actor = DeterministicActor(
            self.state_dim, configs.get("actor_hidden_size"), self.action_dim
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=configs.get("actor_lr")
        )
        # Q1
        self.critic_1 = Critic(
            self.state_dim, configs.get("critic_hidden_size"), self.action_dim
        ).to(self.device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        # Q2
        self.critic_2 = Critic(
            self.state_dim, configs.get("critic_hidden_size"), self.action_dim
        ).to(self.device)
        self.critic_target_2 = copy.deepcopy(self.critic_2)

        self.critic_params = itertools.chain(
            self.critic_1.parameters(), self.critic_2.parameters()
        )
        self.critic_optim = torch.optim.Adam(
            self.critic_params, lr=configs.get("critic_lr")
        )

        self.models = {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "actor_optim": self.actor_optim,
            "critic_1": self.critic_1,
            "critic_target_1": self.critic_target_1,
            "critic_2": self.critic_2,
            "critic_target_2": self.critic_target_2,
            "critic_optim": self.critic_optim,
        }

    def update_param(self, states, actions, rewards, next_states, not_dones):
        with torch.no_grad():
            # select action according to policy and add clipped noise
            noises = (torch.randn_like(actions) * self.sigma).clamp(-self.c, self.c)

            next_actions = (
                self.squash_action(self.actor_target(next_states)) + noises
            ).clamp(-self.action_high, self.action_high)

            # compute the target Q value
            target_Q1, target_Q2 = self.critic_target_1(
                next_states, next_actions
            ), self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_dones * self.gamma * target_Q

        # get current Q estimates
        current_Q1, current_Q2 = self.critic_1(states, actions), self.critic_2(
            states, actions
        )

        # compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # delayed policy updates
        if self.total_it % self.policy_delay == 0:
            self.critic_1.eval(), self.critic_2.eval()

            # compute actor losse
            actor_loss = -torch.mean(
                self.critic_1(states, self.squash_action(self.actor(states)))
            )
            # optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_1.train(), self.critic_2.train()

            # update the frozen target models
            soft_update(self.rho, self.critic_1, self.critic_target_1)
            soft_update(self.rho, self.critic_2, self.critic_target_2)
            soft_update(self.rho, self.actor, self.actor_target)

        return (
            {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
            if self.total_it % self.policy_delay == 0
            else {
                "critic_loss": critic_loss.item(),
            }
        )
