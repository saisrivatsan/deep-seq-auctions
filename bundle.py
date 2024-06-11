import numpy as np

from gymnasium import Env
from gymnasium.spaces import Box


class AuctionEnv(Env):
    def __init__(self, num_agents, num_items, v_dist):
        """ Initialize environment """
        self.num_agents = num_agents
        self.num_items = num_items
        self.allocs = v_dist.allocs
        self.num_menus = v_dist.num_menus
        self.action_scale = v_dist.action_scale
        self.sample = v_dist.sample
        self.action_space = Box(low = 0.0, high = 1.0, shape=(self.num_menus,), dtype=np.float32)
        self.observation_space = Box(low = 0.0,  high = 1.0,  shape=(self.num_items + 1,), dtype=np.float32)
        self.reset()
        
    
    def preprocess_action(self, action):
        """ Scale action and set unavailable bundles price to MAX """
        action = action * self.action_scale
        mask = self.allocs @ (1 - self.state[:self.num_items])
        action[mask > 0] = 1000
        action[0] = 0.0
        return action
    
        
    def step(self, action):
        """ Preprocess action """
        action = self.preprocess_action(action)
        
        """ Sample valuation """
        V = self.sample()
        
        """ Agent picks the utility maximizing option, env computes the reward """
        utility = V - action
        idx = np.argmax(utility)
        reward = action[idx]
                
        """ Update states and remaining items """
        self.state[:self.num_items] = self.state[:self.num_items] - self.allocs[idx]
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
    
    
class SymUnifAuctionEnv(Env):
    def __init__(self, num_agents, num_items):
        """ Initialize environment """
        self.num_agents = num_agents
        self.num_items = num_items
        self.num_menus = num_items + 1
        self.action_space = Box(low = 0.0, high = self.num_items, shape=(self.num_menus,), dtype=np.float32)
        self.observation_space = Box(low = 0.0,  high = 1.0,  shape=(self.num_items + 1,), dtype=np.float32)
        self.reset()

    def step(self, action):
        """ Sample valuation """
        V = np.random.rand(self.num_items)
        V[self.state[:-1] < 1] = -1000
        sort_idx = np.argsort(V)[::-1]

        V = np.insert(np.cumsum(V[sort_idx]), 0, 0)
        utility = V[:len(action)] - action

        sel_idx = np.argmax(utility)
        reward = action[sel_idx]

        allocs = np.zeros(self.num_items)
        allocs[sort_idx[:sel_idx]] = 1.0
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
