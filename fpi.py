import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


""" Fitted Policy Iteration for Combinatorial Menus """
class FPI:
    def __init__(self, envs, agent, rollout_buffer, args, v_dist):
        
        self.envs = envs
        self.agent = agent
        self.rollout_buffer = rollout_buffer
        self.args = args
        
        self.device = args.device
        self.num_steps = self.args.num_agents
        self.v_dist = v_dist

    def collect_rollouts(self):

        self.rollout_buffer.reset()
        obs, _ = self.envs.reset()
        obs = torch.Tensor(obs).to(self.device)
        done = torch.zeros(self.args.num_envs).to(self.device)

        for _ in range(0, self.num_steps):

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(obs)
                value = value.flatten()

            next_obs, reward, terminations, truncations, _ = self.envs.step(action.cpu().numpy())
            reward = torch.tensor(reward).to(self.device).view(-1)
            self.rollout_buffer.add(obs, action, logprob, reward, done, value)

            done = np.logical_or(terminations, truncations)
            obs, done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

        # Compute Values through TD-Learning
        with torch.no_grad():
            next_value = torch.zeros(self.args.num_envs).to(self.device)
            self.rollout_buffer.compute_returns_and_advantages(next_value, done)


    def init_optimizers(self, lr_vf, lr_pi):
        self.opt_vf = torch.optim.Adam(self.agent.critic.parameters(), lr = lr_vf)
        self.opt_pi = torch.optim.Adam(self.agent.actor.parameters(), lr = lr_pi)


    """ Value Function """
    def fit_value(self, vf_epochs):
                
        for epoch in range(vf_epochs):

            value_loss_epoch = 0.0
            self.opt_vf.zero_grad()

            for mb_obs, _, _, _, mb_returns, _ in self.rollout_buffer.get(batch_size = None):

                values = self.agent.get_value(mb_obs).flatten()
                value_loss = F.mse_loss(mb_returns, values)
                value_loss.backward()
                value_loss_epoch += value_loss.item()

            self.opt_vf.step()

    def TD(self):
        raise NotImplementedError

    """ Policy Improvement Step """
    def fit_policy(self, pi_epochs):
        
        tau = self.args.tau
        batch_size = self.args.batch_size
        num_items = self.args.num_items
        num_agents = self.args.num_agents
        n_samples = self.args.num_samples_for_pi
        num_batches = (self.args.num_agents * self.args.num_envs)//batch_size

        # TODO
        alloc = self.v_dist.allocs_tensor
        num_menus = self.v_dist.num_menus
        next_states = torch.zeros((batch_size, num_menus, num_items + 1), device = self.device)
        
        scalers = torch.Tensor(self.v_dist.action_scale).to(device = self.device)


        for epoch in range(pi_epochs):

            self.opt_pi.zero_grad()
            rev_loss_epoch = 0.0

            for mb_obs, _, _, _, _, _ in self.rollout_buffer.get(batch_size = batch_size):

                """ Preprocess action """
                pay = self.agent.get_action(mb_obs) * scalers
                mask = (1 - mb_obs[:, :num_items]) @ alloc.T
                mask = (mask > 0).type(torch.float)
                pay = ((1 - mask) * pay + mask * 100)
                pay[:, 0] = 0

                """ Compute Next State and Offset """
                next_states[..., :num_items] = mb_obs[:, None, :num_items] - alloc[None, :, :]
                next_states[..., -1] = mb_obs[:, None, -1] + 1.0
                next_state_mask = ((next_states < 0).sum(dim = -1) <= 0) * (next_states[..., -1] < num_agents)
                with torch.no_grad():
                    offset = self.agent.get_value(next_states.view(-1, num_items + 1)).flatten()
                    offset = offset.view(batch_size, num_menus)
                    offset = offset * next_state_mask.type(torch.float)

                V_SAMPLES = self.v_dist.sample_tensor(n_samples)
                utilities = V_SAMPLES - pay[:, None, :]
                selector = F.softmax(utilities * tau, dim = -1)

                revenue_loss = -torch.sum(selector * (pay + offset)[:, None, :], dim = -1).mean(-1)
                revenue_loss = revenue_loss.mean()/num_batches
                revenue_loss.backward()
                rev_loss_epoch += revenue_loss.item()

            self.opt_pi.step()
            
    def learn(self):
        self.init_optimizers(self.args.lr_vf, self.args.lr_pi)
        self.collect_rollouts()
        self.fit_value(self.args.vf_epochs)
        self.fit_policy(self.args.pi_epochs)
        self.agent.actor_logstd.data -= self.args.log_std_decay

            
""" Fitted Policy Iteration for Entry-Fee Menus"""            
class FPIScale(FPI):
    
        
    """ TD-Values """
    def TD(self, td_epochs):
        
        self.fit_value(td_epochs)

        batch_size = self.args.batch_size
        num_items = self.args.num_items
        num_agents = self.args.num_agents
        n_samples = self.args.num_samples_for_pi

        for mb_obs, _, _, _, _, _, inds in self.rollout_buffer.get(batch_size = batch_size, return_inds = True):

            with torch.no_grad():
                next_states = torch.zeros((batch_size, n_samples, num_items + 1), device = self.device)

                actions = self.agent.get_action(mb_obs)
                posted_prices, entry_fee = actions[:, :-1], actions[:, -1]
                mask = (mb_obs[:, :num_items] < 1)
                posted_prices[mask] = 1000

                v_samples = torch.rand(n_samples, num_items, device = self.device)
                welfare = v_samples[None, :, :] - posted_prices[:, None, :]
                sort_idx = torch.argsort(-welfare, dim = -1)
                utility = torch.cumsum(torch.gather(welfare, -1, sort_idx), dim = -1) - entry_fee[:, None, None]
                rewards = torch.cumsum(torch.gather(posted_prices[:, None, :].expand(-1, n_samples, -1), -1, sort_idx), dim = -1) + entry_fee[:, None, None]
                rewards = torch.nn.functional.pad(rewards, (1, 0, 0, 0, 0, 0))
                utility = torch.nn.functional.pad(utility, (1, 0, 0, 0, 0, 0))


                alloc = torch.ones(num_items, num_items, device = self.device).triu()
                alloc = alloc[torch.argsort(sort_idx, dim = -1).view(-1, num_items)].transpose(2,1)
                alloc = alloc.view(batch_size, n_samples, num_items, num_items)
                alloc = torch.nn.functional.pad(alloc, (0, 0, 1, 0))

                sel_idx = torch.argmax(utility, dim = -1)
                reward = rewards[torch.arange(batch_size).unsqueeze(1),  torch.arange(n_samples).unsqueeze(0), sel_idx]
                alloc = alloc[torch.arange(batch_size).unsqueeze(1), torch.arange(n_samples).unsqueeze(0), sel_idx, :]

                # Compute next state
                next_states[:, :, :num_items] = mb_obs[:, :num_items][:, None, :] - alloc
                next_states[:, :, -1] = (mb_obs[:, -1] + 1.0)[:, None]
                next_state_mask = ((next_states < 0).sum(dim = -1) <= 0) * (next_states[..., -1] < num_agents)
                next_state_mask = next_state_mask.view(-1, 1) > 0
                idx = torch.where(next_state_mask > 0)[0]
                offset = torch.zeros(next_state_mask.size(0), 1, device = self.device)
                offset[idx] = self.agent.get_value(next_states.view(-1, num_items + 1)[idx])
                offset = offset.view(batch_size, n_samples)

                self.rollout_buffer.returns[inds] = (reward + offset).mean(-1)
                
        
                
                


    """ Policy Improvement Step """
    def fit_policy(self, pi_epochs):
        
        tau = self.args.tau
        batch_size = self.args.batch_size
        num_items = self.args.num_items
        num_agents = self.args.num_agents
        n_samples = self.args.num_samples_for_pi
        num_batches = (self.args.num_agents * self.args.num_envs)//batch_size
        num_menus = num_items + 1
        
        for epoch in range(pi_epochs):

            self.opt_pi.zero_grad()
            rev_loss_epoch = 0.0
            next_states = torch.zeros((batch_size, n_samples, num_menus, num_items + 1), device = self.device)

            for mb_obs, _, _, _, _, _ in self.rollout_buffer.get(batch_size = batch_size):

                """ Preprocess action """
                actions = self.agent.get_action(mb_obs)
                posted_prices, entry_fee = actions[:, :-1], actions[:, -1]
                mask = (mb_obs[:, :num_items] < 1)
                posted_prices[mask] = 1000
                
                V_SAMPLES = self.v_dist.sample_tensor(n_samples)
                welfare = V_SAMPLES[None, :, :] - posted_prices[:, None, :]
                sort_idx = torch.argsort(-welfare, dim = -1)
                utility = torch.cumsum(torch.gather(welfare, -1, sort_idx), dim = -1) - entry_fee[:, None, None]
                rewards = torch.cumsum(torch.gather(posted_prices[:, None, :].expand(-1, n_samples, -1), -1, sort_idx), dim = -1) + entry_fee[:, None, None]
                rewards = torch.nn.functional.pad(rewards, (1, 0, 0, 0, 0, 0))
                utility = torch.nn.functional.pad(utility, (1, 0, 0, 0, 0, 0))
                selector = torch.softmax(utility * tau, dim = -1)
                               
                """ Compute Next State and Offset """
                with torch.no_grad():
                    alloc = torch.ones(num_items, num_items, device = self.device).triu()
                    alloc = alloc[torch.argsort(sort_idx, dim = -1).view(-1, num_items)].transpose(2,1)
                    alloc = alloc.view(batch_size, n_samples, num_items, num_items)
                    alloc = torch.nn.functional.pad(alloc, (0, 0, 1, 0))

                    next_states[:, :, :, :num_items] = mb_obs[:, :num_items][:, None, None, :] - alloc
                    next_states[:, :, :, -1] = (mb_obs[:, -1] + 1.0)[:, None, None]
                    next_state_mask = ((next_states < 0).sum(dim = -1) <= 0) * (next_states[..., -1] < num_agents)
                    next_state_mask = next_state_mask.view(-1, 1) > 0
                    idx = torch.where(next_state_mask > 0)[0]
                    offset = torch.zeros(next_state_mask.size(0), 1, device = self.device)
                    offset[idx] = self.agent.get_value(next_states.view(-1, num_items + 1)[idx])
                    offset = offset.view(batch_size, n_samples, num_menus)
                    
                

                revenue_loss = -torch.sum(selector * (rewards + offset), dim = -1).mean(-1)
                revenue_loss = revenue_loss.mean()/num_batches
                revenue_loss.backward()

            self.opt_pi.step()
            
    def learn(self):
        self.init_optimizers(self.args.lr_vf, self.args.lr_pi)
        self.collect_rollouts()
        
        #self.TD(self.args.vf_epochs)
        self.fit_value(self.args.vf_epochs)
        self.fit_policy(self.args.pi_epochs)
        
        self.agent.actor_logstd.data -= self.args.log_std_decay