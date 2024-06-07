import os
import time
import random
import argparse
import itertools

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import gymnasium as gym
from gymnasium import Env, spaces

from utils import init_logger, evaluate_policy, make_env
import bundle
import entryfee
from buffer import RolloutBuffer
from net import ActorCriticNetworkBundle, ActorCriticNetworkEntryFee
from fpi import FPI, FPIScale
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
    
    lr_vf: float = 1e-3
    lr_pi: float = 1e-4
    """Learning Rate for value fitting and policy improvement"""
    
    vf_epochs: int = 200
    pi_epochs: int = 200
    """ Epochs for value fitting and policy improvement """
    
    batch_size: int = 256
    """ Minibatch size """
    
    num_envs: int = 1024
    """ Number of parallel environments """
        
    gamma: float = 1.0
    """ Discount Factor """
    
    gae_lambda: float = 0.95
    """ GAE lambda """
    
    tau: float = 100
    """ Softmax temperature """
    
    num_samples_for_pi: int = 256
    """ Number of samples to estimate gradient in policy improvement step """
    
    log_std_decay: float = 0.25
    """ How much to decay log_std after every iteration """
    
    max_iteration: int = 20
    """ Max iteration """

    
    """ Miscellaneous Params --------------------- """
    
    device: str = "cuda"
    """ CUDA or CPU """
    
    t_max: int = 8 * 60 * 60
    """ Max time to train: 8 hrs """
    
    print_iter: int = 100
    """ When to log stats """
    
    seed: int = 24
    """seed of the experiment"""


if __name__ == "__main__":     
    
    """ Parse arguments """
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--num_agents', action='store',
                        dest='num_agents', required=True, type=int,
                        help='Num Agents')

    parser.add_argument('-m', '--num_items', action='store',
                        dest='num_items', required=True, type=int,
                        help='Num Items')
    
    parser.add_argument('-e', '--env_type', action='store',
                        dest='env_type', required=True, type=str,
                        help='Env Type')


    cmd_args = parser.parse_args()

    """ Set hyper-params """
    args = Args()
    args.num_agents = cmd_args.num_agents
    args.num_items = cmd_args.num_items
    args.env_type = cmd_args.env_type
        
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
        model_class = FPI

    elif args.env_type == "unif":
        v_dist = UNIFScale(args.num_items, demand = None)
        args.pi_epochs = 50

        env_class = entryfee.AuctionEnv
        policy_class = ActorCriticNetworkEntryFee
        model_class = FPIScale
        

    else:
        print("Auction Env not supported")
        exit(1)

    v_dist.set_action_scale([1.0])


    """ Loggers """
    log_fname = os.path.join("experiments", "FPI", "%s_%dx%d"%(args.env_type, args.num_agents, args.num_items))
    logger = init_logger(log_fname)


    """ Seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv([make_env(env_class, args.num_agents, args.num_items, v_dist) for i in range(args.num_envs)])
    eval_envs = gym.vector.SyncVectorEnv([make_env(env_class, args.num_agents, args.num_items, v_dist) for i in range(args.num_envs)])
    agent = policy_class(envs, args.num_hidden_layers, args.num_hidden_units, args.d_model, args.num_agents + 1, args.log_std_init).to(args.device)
    rollout_buffer = RolloutBuffer(envs, args.num_agents, args.gamma, args.gae_lambda, args.device)
    model = model_class(envs, agent, rollout_buffer, args, v_dist)   


    """ Train """
    tic = time.time()
    for iteration in range(args.max_iteration):
        model.learn()
        t = time.time() - tic
        rev_eval = evaluate_policy(agent, envs, num_eval_episodes = 10240)
        logger.info("[Iter]: %d, [Time Elapsed]: %.4f, [Rev]: %.6f"%(iteration + 1, t, rev_eval))