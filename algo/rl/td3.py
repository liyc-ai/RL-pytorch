import torch
import torch.nn.functional as F
import copy
import numpy as np
import itertools
from algo.rl.base import BaseAgent
from network.actor import DeterministicActor
from network.critic import Critic
from utils.buffer import SimpleReplayBuffer
from utils.net import soft_update

class TD3Agent(BaseAgent):
    """Twin Delayed Deep Deterministic Policy Gradient
    
    Code modified from: https://github.com/sfujim/TD3
    """
    def __init__(self, configs):
        super().__init__(configs)
        self.policy_delay = configs['policy_delay']
        self.rho = configs['rho']
        
        # noise injection
        self.c = configs['c']*self.action_high
        self.sigma = configs['sigma']*self.action_high
        self.expl_std = configs['expl_std']*self.action_high
        self.total_it = 0
        
        self.actor = DeterministicActor(self.state_dim, configs['actor_hidden_size'], self.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=configs['actor_lr'])
        
        # Q1
        self.critic_1 = Critic(self.state_dim, configs['critic_hidden_size'], self.action_dim).to(self.device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        # Q2
        self.critic_2 = Critic(self.state_dim, configs['critic_hidden_size'], self.action_dim).to(self.device)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        
        self.critic_params = itertools.chain(self.critic_1.parameters(), self.critic_2.parameters())
        self.critic_optim = torch.optim.Adam(self.critic_params, lr=configs['critic_lr'])
        
        self.replay_buffer = SimpleReplayBuffer(self.state_dim, self.action_dim, self.device, configs['buffer_size'])
        
    def select_action(self, state, training=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.transform_action(self.actor(state)).cpu().data.numpy().flatten()
            if training:
                action = (action
                    + np.random.normal(0, self.expl_std, size=self.action_dim)
                ).clip(-self.action_high, self.action_high)
        return action
    
    def update(self, state, action, next_state, reward, done):
        self.total_it += 1
        self.replay_buffer.add(state, action, next_state, reward, done)
        
        if self.replay_buffer.size < self.configs['start_timesteps']: return
        
        
        # sample replay buffer 
        states, actions, next_states, rewards, not_dones = self.replay_buffer.sample(self.configs['batch_size'])

        with torch.no_grad():
            # select action according to policy and add clipped noise
            noises = (
                torch.randn_like(actions) * self.sigma
            ).clamp(-self.c, self.c)
            
            next_actions = (
                self.transform_action(self.actor_target(next_states)) + noises
            ).clamp(-self.action_high, self.action_high)

            # compute the target Q value
            target_Q1, target_Q2 = self.critic_target_1(next_states, next_actions), self.critic_target_2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + not_dones * self.gamma * target_Q

        # get current Q estimates
        current_Q1, current_Q2 = self.critic_1(states, actions), self.critic_2(states, actions)
        
        # compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # delayed policy updates
        if self.total_it % self.policy_delay == 0:
            self.critic_1.eval(), self.critic_2.eval()
            
            # compute actor losse
            actor_loss = -torch.mean(self.critic_1(states, self.transform_action(self.actor(states))))
            # optimize the actor 
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            self.critic_1.train(), self.critic_2.train()

            # update the frozen target models
            soft_update(self.rho, self.critic_1, self.critic_target_1)
            soft_update(self.rho, self.critic_2, self.critic_target_2)
            soft_update(self.rho, self.actor, self.actor_target)
        