import os
import time
import random
import logging
import argparse
import itertools

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy


import bundle
import entryfee
from utils import init_logger
from distribution import UNIF, ASYM, COMB1, COMB2, UNIFScale

""" Hyper-parameters """
class Args:
    
    
    """ Env Params ------------------------------ """
    
    env_type: str = ""
    """ Environment Type (To be filled later) """

    num_agents: int = 0
    """Number of agents (To be filled later) """

    num_items: int = 0
    """Number of items (To be filled later) """
    
    
    """ Policy Params ---------------------------- """
    
    log_std_init: float = -2
    """std for exploration"""
    
    num_hidden_units: int = 256
    """ Number of hidden units"""
    
    num_hidden_layers: int = 3
    """ Number of hidden layers """
    
    d_model: int = 12
    """ Positional Embedding Dimensions """
    
    
    """ Optimization Params ---------------------- """
    
    learning_rate: float = 3e-4
    """Learning Rate """
    
    batch_size: int = 1024
    """ Minibatch size """
    
    num_envs: int = 1024
    """ Number of parallel environments """
        
    gamma: float = 1.0
    """ Discount Factor """
     
 
    """ Miscellaneous Params --------------------- """
    
    device: str = "cuda"
    """ CUDA or CPU """
        
    max_iteration: int = 20000
    """ Max iteration """
    
    print_iter: int = 100
    """ When to log stats """
    
    seed: int = 24
    """seed of the experiment"""


""" Positional Encoding Feature Extractor """
class CustomFeatureExtractor(BaseFeaturesExtractor):
    
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    
    def __init__(self, observation_space, d_model, max_len):
        
        
        self._observation_space = observation_space
        self._features_dim = get_flattened_obs_dim(observation_space) - 1 + d_model
        super().__init__(observation_space, self._features_dim)
        self.flatten = nn.Flatten()        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    @property
    def features_dim(self):
        return self._features_dim
    
    def forward(self, obs):
        obs = self.flatten(obs)
        agent_idx = (obs[:, -1]).type(torch.long)
        pe_features = self.pe[agent_idx]
        return torch.cat((obs[:, :-1], pe_features), dim = -1)
    

class ActorCriticNetworkBundle(ActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        sigmoid_offset = 0.5
        mean_actions = torch.sigmoid(self.action_net(latent_pi) - sigmoid_offset)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
    
class ActorCriticNetworkEntryFee(ActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        sigmoid_offset = 0.5
        softplus_offset = 1.0
        
        mean_actions = self.action_net(latent_pi)
        posted_price = F.sigmoid(mean_actions[..., :-1] - sigmoid_offset)
        entry_fee = F.softplus(mean_actions[..., -1:] - softplus_offset)
        mean_actions = torch.cat([posted_price, entry_fee], dim = -1)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
    

if __name__ == "__main__":     
    
    """ Parse arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--num_agents', action='store',
                        dest='num_agents', required=True, type=int,
                        help='Num Agents')

    parser.add_argument('-m', '--num_items', action='store',
                        dest='num_items', required=True, type=int,
                        help='Num Items')
    
    parser.add_argument('-l', '--learning_rate', action='store',
                    dest='lr', required=True, type=float,
                    help='Learning Rate')

    parser.add_argument('-e', '--env_type', action='store',
                        dest='env_type', required=True, type=str,
                        help='Env Type')


    cmd_args = parser.parse_args()
    
    """ Set hyper-params """
    args = Args()
    args.num_agents = cmd_args.num_agents
    args.num_items = cmd_args.num_items
    args.env_type = cmd_args.env_type
    args.learning_rate = cmd_args.lr
    
    
    """ Environment Type """
    if args.num_items <= 10:
        if args.env_type == "unif":
            v_dist = UNIF(args.num_items, demand = None)

        elif args.env_type == "unit":
            v_dist = UNIF(args.num_items, demand = 1)

        elif args.env_type == "3demand":
            v_dist = UNIF(args.num_items, demand = 3)

        elif args.env_type == "asym":
            v_dist = ASYM(args.num_items, demand = None)

        elif args.env_type == "comb1":
            v_dist = COMB1(args.num_items, demand = None)

        elif args.env_type == "comb2":
            v_dist = COMB2(args.num_items, demand = None)

        else:
            print("Auction Env not supported")
            exit(1)
            
        env_class = bundle.AuctionEnv
        policy_class = ActorCriticNetworkBundle
            
                      
    elif args.env_type == "unif":
        v_dist = UNIFScale(args.num_items, demand = None)
        env_class = entryfee.AuctionEnv
        policy_class = ActorCriticNetworkEntryFee
       
    else:
        print("Auction Env not supported")
        exit(1)
        
        
    if args.num_agents <= 5:
        args.max_iteration = 10000
        
    """ Loggers """
    log_fname = os.path.join("experiments", "PPO", "%s_%dx%d"%(args.env_type, args.num_agents, args.num_items))
    logger = init_logger(log_fname)


    """ Seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
        
        
    env_kwargs = {'num_agents': args.num_agents, 'num_items': args.num_items, 'v_dist': v_dist}
    envs = make_vec_env(env_class, args.num_envs, env_kwargs = env_kwargs)
    eval_env = make_vec_env(env_class, args.num_envs, env_kwargs = env_kwargs)

        
    """ Set policy params """
    policy_kwargs = dict(
        log_std_init = args.log_std_init,
        net_arch = dict(pi=[args.num_hidden_units] * args.num_hidden_layers, vf=[args.num_hidden_units] * args.num_hidden_layers),
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(d_model = args.d_model, max_len = args.num_agents + 1),
    )

    """ Init PPO model """
    model = PPO(policy_class,
            envs,
            learning_rate = args.learning_rate,
            n_steps = args.num_agents,
            batch_size = args.batch_size,
            gamma = args.gamma,
            policy_kwargs = policy_kwargs)

    """ Training """
    tic = time.time()
    for iteration in range(args.max_iteration):
        model.learn(args.num_envs * args.num_agents)
        
        if iteration % args.print_iter == 0:
            t = time.time() - tic
            rev_eval = evaluate_policy(model, eval_env, n_eval_episodes=10240)[0]
            logger.info("[Iter]: %d, [Time Elapsed]: %.4f, [Rev]: %.6f"%(iteration, t, rev_eval))
            
    t = time.time() - tic
    rev_eval = evaluate_policy(model, eval_env, n_eval_episodes=10240)[0]
    logger.info("[Iter]: %d, [Time Elapsed]: %.4f, [Rev]: %.6f"%(iteration, t, rev_eval))

            
