import os
import logging

import torch

""" Loggers """
def init_logger(log_fname):
    
    root_dir = os.path.dirname(log_fname)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler(log_fname, 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


""" Evaluate policy """
def evaluate_policy(agent, envs, num_eval_episodes = 10240):
    agent.eval()
    device = next(agent.parameters()).device
    returns = 0.0
    num_eval_batches = int(num_eval_episodes/envs.num_envs)
    for _ in range(num_eval_batches):
        terminated = False
        obs, _ = envs.reset()
        while not terminated:
            with torch.no_grad():
                actions = agent.get_action(torch.Tensor(obs).to(device))
            next_obs, reward, terminated, _, _ = envs.step(actions.cpu().numpy())
            terminated = terminated[0] # all envs have the same number of steps
            returns += reward.mean()
            obs = next_obs
    return returns/num_eval_batches

""" Helper to create env """
def make_env(env_class, *args):
    def thunk():
        env = env_class(*args)
        return env

    return thunk