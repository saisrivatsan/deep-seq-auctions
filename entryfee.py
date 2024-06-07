import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box


class AuctionEnv(Env):
    def __init__(self, num_agents, num_items, v_dist):
        """ Initialize environment """
        self.num_agents = num_agents
        self.num_items = num_items
        self.action_scale = v_dist.action_scale
        self.sample = v_dist.sample
        self.action_space = Box(low = 0.0, high = self.num_items, shape=(self.num_items + 1,), dtype=np.float32)
        self.observation_space = Box(low = 0.0,  high = 1.0,  shape=(self.num_items + 1,), dtype=np.float32)
        self.reset()
        
    
    def preprocess_action(self, action):
        """ Scale action and set unavailable bundles price to MAX """        
        posted_prices, entry_fee = action[:-1], action[-1]
        mask = self.state[:self.num_items] < 1
        posted_prices[mask] = 1000
        return posted_prices, entry_fee
    
        
    def step(self, action):
        """ Preprocess action """
        posted_prices, entry_fee = self.preprocess_action(action)
        
        """ Sample valuation """
        V = self.sample()
        
        """ Agent picks the utility maximizing option, env computes the reward """
        welfare = V - posted_prices
        sort_idx = np.argsort(-welfare)
        
        utility = np.cumsum(welfare[sort_idx]) - entry_fee
        rewards = np.cumsum(posted_prices[sort_idx]) + entry_fee
        
        rewards = np.insert(rewards, 0, 0)
        utility = np.insert(utility, 0, 0)
        
        sel_idx = np.argmax(utility)
        reward = rewards[sel_idx]
        allocs = np.zeros(self.num_items)
        allocs[sort_idx[:sel_idx]] = 1.0
                
        """ Update states and remaining items """
        self.state[:self.num_items] = self.state[:self.num_items] - allocs
        self.state[-1] += 1.0
        
        """ Check if done """
        terminated = (int(self.state[-1]) == self.num_agents)
        
        return self.state, reward, terminated, False, {}

    def reset(self, seed = None, options = None):
        """ Reset environment to original state """
        super().reset(seed=seed) 
        self.state = np.ones(self.num_items + 1, dtype = np.float32)
        self.state[-1] = 0.0
        return self.state, {}
