import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal


""" Positional Encoding Feature Extractor """
class CustomFeatureExtractor(nn.Module):
    
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    
    def __init__(self, obs_dim, d_model, max_len):
        super().__init__()
        
        self.obs_dim = obs_dim
        self._features_dim = obs_dim - 1 + d_model    
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
        agent_idx = (obs[:, -1]).type(torch.long)
        pe_features = self.pe[agent_idx]
        return torch.cat((obs[:, :-1], pe_features), dim = -1)

""" Offset Softplus """
class SoftplusOffset(nn.Module):
    def __init__(self, offset = 1.0):
        super().__init__()
        self.offset = offset

    def forward(self, input):
        return F.softplus(input) - self.offset
    

class SoftplusAndSigmoid(nn.Module):
    def __init__(self, offset_softplus = 1.0, offset_sigmoid = 0.5):
        super().__init__()
        self.offset_sigmoid = offset_sigmoid
        self.offset_softplus = offset_softplus

    def forward(self, input):
        posted_price = F.sigmoid(input[..., :-1] - self.offset_sigmoid)
        entry_fee = F.softplus(input[..., -1:] - self.offset_softplus)
        return torch.cat([posted_price, entry_fee], dim = -1)
    
    
""" ActorCritic Network """
class ActorCriticNetworkBundle(nn.Module):
    def __init__(self, envs, num_hidden_layers, num_hidden_units, d_model, max_len, log_std_init):
        
        super().__init__()
        
        feature_extractor = CustomFeatureExtractor(envs.single_observation_space.shape[0], d_model, max_len)
        
        policy_net = [feature_extractor]
        value_net = [feature_extractor]
        
        last_layer_dim = feature_extractor.features_dim
        for _ in range(num_hidden_layers):
            policy_net.append(nn.Linear(last_layer_dim, num_hidden_units))
            policy_net.append(nn.Tanh())
            value_net.append(nn.Linear(last_layer_dim, num_hidden_units))
            value_net.append(nn.Tanh())
            last_layer_dim = num_hidden_units
            
        value_net.append(nn.Linear(last_layer_dim, 1))
        policy_net.append(nn.Linear(last_layer_dim, np.prod(envs.single_action_space.shape)))
        policy_net.append(SoftplusOffset())
        
        self.critic = nn.Sequential(*value_net)
        self.actor = nn.Sequential(*policy_net)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.actor_logstd.data += log_std_init

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x):
        return self.actor(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.get_action(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

class ActorCriticNetworkEntryFee(ActorCriticNetworkBundle):
    def __init__(self, envs, num_hidden_layers, num_hidden_units, d_model, max_len, log_std_init):
        
        super().__init__(envs, num_hidden_layers, num_hidden_units, d_model, max_len, log_std_init)
        
        feature_extractor = CustomFeatureExtractor(envs.single_observation_space.shape[0], d_model, max_len)
        
        policy_net = [feature_extractor]
        value_net = [feature_extractor]
        
        last_layer_dim = feature_extractor.features_dim
        for _ in range(num_hidden_layers):
            policy_net.append(nn.Linear(last_layer_dim, num_hidden_units))
            policy_net.append(nn.Tanh())
            value_net.append(nn.Linear(last_layer_dim, num_hidden_units))
            value_net.append(nn.Tanh())
            last_layer_dim = num_hidden_units
            
        value_net.append(nn.Linear(last_layer_dim, 1))
        policy_net.append(nn.Linear(last_layer_dim, np.prod(envs.single_action_space.shape)))
        policy_net.append(SoftplusAndSigmoid())
        
        self.critic = nn.Sequential(*value_net)
        self.actor = nn.Sequential(*policy_net)
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.actor_logstd.data += log_std_init