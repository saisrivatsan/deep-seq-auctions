import os
import time
import random
import argparse
import itertools

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from bundle import AuctionEnv
from utils import init_logger
from distribution import UNIF, ASYM,  COMB1, COMB2


""" Hyper-parameters """
class Args:
       
    """ Env Params ------------------------------ """
    
    env_type: str = ""
    """ Environment Type (To be filled later) """

    num_agents: int = 0
    """Number of agents (To be filled later) """

    num_items: int = 0
    """Number of items (To be filled later) """
    
    
    """ Optimization Params ---------------------- """
    
    learning_rate: float = 1e-2
    """Learning Rate for RochetNet"""
    
    max_iter: int = 2000
    """ Num iterations to train """
    
    batch_size: int = 2**15
    """ Minibatch size """
    
    num_val_batches: int = 100
    """ Number of samples to compute offset """
        
    gamma: float = 1.0
    """ Discount Factor """
    
    tau: float = 100
    """ Softmax temperature """
    

    """ Miscellaneous Params --------------------- """
    
    device: str = "cuda"
    """ CUDA or CPU """
            
    seed: int = 24
    """seed of the experiment"""
    
    
class RochetNetOffsets(nn.Module):      
    def __init__(self, allocs, offsets, tau = 100, scale = 1.0):
        super().__init__()
        self.num_menus, self.num_items = allocs.shape
        self.register_buffer("allocs", torch.Tensor(allocs))
        self.register_buffer("offsets", torch.Tensor(offsets))
        
        self.pay = nn.Parameter(torch.Tensor(self.num_menus))
        
        self.scale = scale
        self.tau = tau
        self.reset_parameters()
        self.null_idx = np.where(allocs.sum(-1) == 0)[0][0]

    def reset_parameters(self):
        """ Intialize paramters """
        nn.init.zeros_(self.pay)
        
    def preprocess_action(self):
        """ Scale actions and ensure IR """
        pay = self.pay * self.scale
        pay[self.null_idx] *= 0
        return pay
        
    def forward(self, x):    
        """
        Compute pay_trn
        Arguments:
            x: [num_instances, num_items]
        Returns:
            pay: [num_instances]
        """     
        
        pay = self.preprocess_action()
        utility = x - pay[None, :]
        return F.softmax(utility * self.tau, dim = -1) @ (pay + self.offsets)
        
    def get_revenue(self, x):
        pay = self.preprocess_action()
        utility = x - pay[None, :]
        menu_idx = torch.argmax(utility, -1)
        return (pay[menu_idx] + self.offsets[menu_idx]).mean()
    
    
def train_rochetnet(allocs, 
                    offset,
                    torch_sampler,
                    lr = 0.01, 
                    max_iter = 10000, 
                    batch_size = 2**15,
                    num_val_batches = 100,
                    tau = 100, 
                    scale = 1.0,
                    device = "cuda"):
    
    
    m = RochetNetOffsets(allocs, offset, tau, scale).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    
    """ Train RochetNet """    
    for i in range(max_iter):    
        opt.zero_grad()
        x = torch_sampler(batch_size)
        loss = -m(x).mean()
        loss.backward()
        opt.step()
            
    
    """ Compute Offset for future """
    with torch.no_grad():
        val_rev = 0.0
        for j in range(num_val_batches):
            x = torch_sampler(batch_size)
            val_rev += m.get_revenue(x)
                
    return m.preprocess_action().detach().cpu().numpy(), val_rev.item()/num_val_batches

def evaluate_policy_numpy(agent, envs, num_eval_episodes = 10240):
    returns = 0.0
    for _ in range(num_eval_episodes):
        terminated = False
        obs, _ = env.reset()
        while not terminated:
            actions = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = envs.step(actions)
            returns += reward
            obs = next_obs
    return returns/num_eval_episodes

    
def unif_sampler(batch_size, allocs, device):
    num_items = allocs.shape[-1]
    v = torch.rand(batch_size, num_items, device = device)
    return v @ allocs.T

def asym_sampler(batch_size, allocs, device):
    num_items = allocs.shape[-1]
    value_scalers_tensor = torch.arange(1, num_items + 1, device = device)/num_items
    v = torch.rand(batch_size, num_items, device = device) * value_scalers_tensor[None, :]
    return v @ allocs.T

def comb1_sampler(batch_size, allocs, device):
    num_menus = allocs.shape[0]
    v = torch.rand(batch_size, num_menus, device = device) * torch.sqrt(allocs.sum(-1))[None, :]
    return v

def comb2_sampler(batch_size, allocs, device):
    num_menus, num_items = allocs.shape
    v = torch.rand(batch_size, num_items, device = device) + 1.0
    c = torch.rand(batch_size, num_menus, device = device) * allocs.sum(-1)[None, :]
    return v @ allocs.T + c

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
    if args.env_type == "unif":
        sampler = unif_sampler
        demand = None
        v_dist = UNIF(args.num_items, demand = None)
        sampler = unif_sampler
        
    elif args.env_type == "unit":
        sampler = unif_sampler
        demand = 1
        v_dist = UNIF(args.num_items, demand = 1)
        sampler = unif_sampler
        

    elif args.env_type == "3demand":
        sampler = unif_sampler
        demand = 3
        v_dist = UNIF(args.num_items, demand = 3)
        sampler = unif_sampler
       

    elif args.env_type == "asym":
        sampler = asym_sampler
        demand = None
        v_dist = ASYM(args.num_items, demand = None)
        sampler = asym_sampler
       
        
    elif args.env_type == "comb1":
        sampler = None
        demand = None
        v_dist = COMB1(args.num_items, demand = None)
        sampler = comb1_sampler
        
    elif args.env_type == "comb2":
        sampler = None
        demand = None
        v_dist = COMB2(args.num_items, demand = None)
        sampler = comb2_sampler
        
    else:
        print("Auction Env not supported")
        exit(1)
        
    
    
    """ Loggers """
    log_fname = os.path.join("experiments", "DP", "%s_%dx%d"%(args.env_type, args.num_agents, args.num_items))
    logger = init_logger(log_fname)


    """ Seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

 
    tic = time.time()
    allocs_all = np.array(list(itertools.product([0, 1], repeat=args.num_items)))
    offsets_all = np.zeros(allocs_all.shape[0])
    actions_all = np.ones((args.num_agents, allocs_all.shape[0], allocs_all.shape[0])) * 1000

    def get_state_idx(state): return int("".join(str(int(c)) for c in state), 2)



    for agent_idx in range(args.num_agents - 1, -1, -1):

        offset_new = np.zeros(allocs_all.shape[0])
        for state in allocs_all:

            item_idx = get_state_idx(state)

            if state.sum() > 0:
                mask = allocs_all @ (1 - state)
                allocs = allocs_all[mask <= 0]
                next_state_idx = list(map(get_state_idx, state[None, :] - allocs))
                offsets = offsets_all[next_state_idx]

                allocs_tensor = torch.tensor(allocs, dtype = torch.float32, device = args.device)
                torch_sampler = lambda batch_size: sampler(args.batch_size, allocs_tensor, args.device)
                scale = v_dist.action_scale_tensor[mask<=0]

                rochetnet_kwargs = dict(lr = args.learning_rate, 
                                        max_iter = args.max_iter, 
                                        batch_size = args.batch_size,
                                        num_val_batches = args.num_val_batches,
                                        tau = args.tau, 
                                        scale = scale,
                                        device = args.device)

                actions, offset_new[item_idx] = train_rochetnet(allocs, offsets, torch_sampler, **rochetnet_kwargs)
                actions_all[agent_idx][item_idx][mask <= 0] = actions

            log_str = "[Agent]: %d, [State]: ["
            for st in range(args.num_items -1): log_str += "%d, "
            log_str += "%d], [Revenue]: %.4f" 
            logger.info(log_str%(agent_idx, *state, offset_new[item_idx]))
        offsets_all = offset_new

   
    class Agent:
        def __init__(self, num_items, v_dist, actions_all):
            super().__init__()
            self.num_items = num_items
            self.actions_all = actions_all
            self.allocs = v_dist.allocs
            self.num_menus = self.allocs.shape[-1]

        def get_action(self, state):
            agent_idx, item_idx = int(state[-1]), get_state_idx(state[:-1])
            return self.actions_all[agent_idx, item_idx]
                
    v_dist.set_action_scale([1.0])
    env = AuctionEnv(args.num_agents, args.num_items, v_dist)
    agent = Agent(args.num_items, v_dist, actions_all)
    t = time.time() - tic
    logger.info("[Time Elapsed]: %.4f, [Test Revenue]: %.4f"%(t, evaluate_policy_numpy(agent, env, num_eval_episodes = 10240)))