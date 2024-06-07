import numpy as np
import torch

""" Rollout Buffer """
class RolloutBuffer:
    def __init__(self, envs, num_steps, gamma, gae_lambda, device = "cuda"):
        self.envs = envs
        self.num_steps = num_steps
        self.num_envs = envs.num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.advantages = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.returns = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        self.pos = 0
        self.full = False
        self.generator_ready = False

    def size(self):
        return self.pos

    def add(self, obs, actions, logprobs, rewards, dones, values):
        self.obs[self.pos] = obs
        self.actions[self.pos] = actions
        self.logprobs[self.pos] = logprobs
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.values[self.pos] = values
        self.pos += 1
        if self.pos == self.num_steps - 1:
            self.full = True

    def get(self, batch_size = None, return_inds = False):
        assert self.full
        indices = np.random.permutation(self.num_steps * self.num_envs)
        if not self.generator_ready:
            self.obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            self.actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            self.logprobs = self.logprobs.reshape(-1)
            self.advantages = self.advantages.reshape(-1)
            self.returns = self.returns.reshape(-1)
            self.values = self.values.reshape(-1)
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.num_steps * self.num_envs

        start_idx = 0
        while start_idx < self.num_steps * self.num_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size], return_inds = return_inds)
            start_idx += batch_size


    def _get_samples(self, batch_inds, return_inds):
        if not return_inds:
            return self.obs[batch_inds], self.actions[batch_inds], self.logprobs[batch_inds], self.advantages[batch_inds], self.returns[batch_inds], self.values[batch_inds]
        else:
            return self.obs[batch_inds], self.actions[batch_inds], self.logprobs[batch_inds], self.advantages[batch_inds], self.returns[batch_inds], self.values[batch_inds], batch_inds


    def compute_returns_and_advantages(self, last_value, done):
        last_value = last_value.detach()
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values   