import numpy as np
import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from algo.rl.base import BaseAgent
from network.actor import StochasticActor
from network.critic import Critic
from utils.buffer import SimpleReplayBuffer
from utils.gae import GAE

class TRPOAgent(BaseAgent):
    """Trust Region Policy Optimization
    """
    def __init__(self, configs):
        super().__init__(configs)
        
        self.rollout_steps = configs['rollout_steps']
        self.lambda_ = configs['lambda']
        self.residual_tol = configs['residual_tol']
        self.cg_steps = configs['cg_steps']
        self.damping = configs['damping']
        self.delta = configs['delta']
        self.beta = configs['beta']
        self.max_backtrack = configs['max_backtrack']
        self.line_search_accept_ratio = configs['line_search_accept_ratio']
        self.n_critic_update = configs['n_critic_update']
        
        self.gae = GAE(self.gamma, self.lambda_)
        self.replay_buffer = SimpleReplayBuffer(self.state_dim, self.action_dim, self.device, int(configs['buffer_size']))
        self.actor = StochasticActor(self.state_dim, configs['actor_hidden_size'], self.action_dim, init=True).to(self.device)
        self.critic = Critic(self.state_dim, configs['critic_hidden_size'], init=True).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), configs['lr'], weight_decay=configs['weight_decay'])
        
    def _conjugate_gradient(self, Hvp_func, g):
        """To calculate s = H^{-1}g without solving inverse of H
        
        Code modified from: https://github.com/ikostrikov/pytorch-trpo
        """
        x = torch.zeros_like(g)
        r = g.clone()
        p = g.clone()
        rdotr = torch.dot(r, r)
        for _ in range(self.cg_steps):
            _Hvp = Hvp_func(p)
            alpha = rdotr / torch.dot(p, _Hvp)
            x += alpha * p
            r -= alpha * _Hvp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < self.residual_tol:
                break
        return x
    
    def _calcu_surrogate_loss(self, log_action_probs):
        return torch.mean(torch.exp(log_action_probs - self.fixed_log_action_probs) * self.advantages)
    
    def _calcu_sample_average_kl(self, mus, stds):
        action_distribution = Normal(mus, stds)
        return torch.mean(kl_divergence(self.fixed_action_distribution, action_distribution))  # sample average kl-divergence
    
    def _line_search(self, update_dir, full_step_size, check_constraints_satisfied):
        """https://en.wikipedia.org/wiki/Backtracking_line_search
        """
        alpha = full_step_size / self.beta

        for _ in range(self.max_backtrack):
            alpha *= self.beta
            if check_constraints_satisfied(alpha * update_dir, alpha):
                return alpha

        return 0.
        
    def _apply_update(self, update):
        """Apply update to actor
        
        Code modified from: torch.nn.utils.convert_parameters.vector_to_parameters
        """
        n = 0
        for param in self.actor.parameters():
            numel = param.numel()
            param_update = update[n:n + numel].view(param.size())
            param.data += param_update
            n += numel
            
    def _Hvp_func(self, v):
            gvp = torch.sum(self.grads*v)
            Hvp = parameters_to_vector(grad(gvp, self.actor.parameters(), retain_graph=True)).clone().detach()
            Hvp += self.damping*v
            return Hvp
            
    def select_action(self, state, training=False):
        state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():  # action follows normal distribution
            action_mean, action_std = self.actor(state)
            if training:
                action = torch.normal(action_mean, action_std)
            else:
                action = action_mean
        return action.cpu().data.numpy().flatten()
        
    def update(self, state, action, next_state, reward, done):
        # collect transitions
        self.replay_buffer.add(state, action, next_state, reward, done)
        if self.replay_buffer.size < self.rollout_steps: return
        
        # estimate advantage
        states, actions, next_states, rewards, not_dones = self.replay_buffer.sample()
        with torch.no_grad():
            Rs, self.advantages = self.gae(self.critic, states, rewards, not_dones, next_states)
            
        # estimate actor gradient
        action_mean, action_std = self.actor(states)
        action_distribution = Normal(action_mean, action_std)
        log_action_probs = action_distribution.log_prob(actions).sum(axis=-1, keepdims=True)  # sum(axis=-1, keepdims=True)
        
        self.fixed_action_distribution = Normal(action_mean.clone().detach(), action_std.clone().detach())
        self.fixed_log_action_probs = self.fixed_action_distribution.log_prob(actions).sum(axis=-1, keepdims=True)
        
        loss = self._calcu_surrogate_loss(log_action_probs)
        g = parameters_to_vector(grad(loss, self.actor.parameters(), retain_graph=True)).clone().detach() # flatten g into a single vector
        
        # Hessian vector product estimation
        kl = self._calcu_sample_average_kl(action_mean, action_std)
        self.grads = parameters_to_vector(grad(kl, self.actor.parameters(), create_graph=True))
        
        update_dir = self._conjugate_gradient(self._Hvp_func, g)  # update direction
        Hvp = self._Hvp_func(update_dir)
        full_step_size = torch.sqrt(2*self.delta/torch.dot(update_dir, Hvp))  # expected update size
        
        # line search
        expected_improvement = torch.dot(g, update_dir)
        def check_constrained(update, alpha):
            with torch.no_grad():
                self._apply_update(update)
                new_action_mean, new_action_std = self.actor(states)
                try:
                    new_action_distribution = Normal(new_action_mean, new_action_std)
                except:
                    raise ValueError("Invalid Gradient!")
                new_log_action_probs = new_action_distribution.log_prob(actions).sum(axis=-1, keepdims=True)
                
                new_loss = self._calcu_surrogate_loss(new_log_action_probs)
                new_mean_kl = self._calcu_sample_average_kl(new_action_mean, new_action_std)
                self._apply_update(-update)

            actual_improvement = new_loss - loss
            improvement_ratio = actual_improvement / (expected_improvement * alpha)
            surrogate_cond = improvement_ratio >= self.line_search_accept_ratio and actual_improvement > 0.0
            
            kl_cond = new_mean_kl <= self.delta
            
            return surrogate_cond and kl_cond
        
        real_step_size = self._line_search(update_dir, full_step_size, check_constrained)
        self._apply_update(real_step_size*update_dir)  # update actor
            
        # update critic
        for _ in range(self.n_critic_update):
            values = self.critic(states)
            critic_loss = F.mse_loss(Rs, values)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
        
        # clear buffer
        self.replay_buffer.clear()
        