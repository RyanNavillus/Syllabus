import torch
import torch.nn as nn
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .procgen_model import SmallNetBase, ResNetBase, MLPBase, Categorical

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, num_actions, use_sde=False, arch='small', base_kwargs=None, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.lr_schedule = lr_schedule  
        self.use_sde = use_sde          
        self.num_actions = num_actions  

        if base_kwargs is None:
            base_kwargs = {}

        if len(observation_space.shape) == 3:
            if arch == 'small':
                self.network = SmallNetBase(observation_space.shape[2], **base_kwargs)
            else:
                self.network = ResNetBase(observation_space.shape[2], **base_kwargs)
        elif len(observation_space.shape) == 1:
            self.network = MLPBase(observation_space.shape[2], **base_kwargs)

        self.dist = Categorical(self.network.output_size, num_actions)
        self.latent_dim_pi = 256
        self.latent_dim_vf = 256

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.network.recurrent_hidden_state_size

    def forward(self, *args, deterministic=False, **kwargs):
        input = args[0].shape[2]
        value, actor_features = self.network(input)
        dist = self.dist(actor_features)
        action = dist.mode() if deterministic else dist.sample()
        action_log_dist = dist.logits
        return action, value, action_log_dist
    
    def get_value(self, input):
        value, _, _ = self.network(input)
        return value
    
    def evaluate_actions(self, input, rnn_hxs, masks, action):
        value, actor_features = self.network(input, rnn_hxs, masks)
        dist = self.dist(actor_features)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy

    
class Sb3ProcgenAgent(CustomPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, num_actions, use_sde=False, arch='large', base_kwargs=None, **kwargs):
        super(Sb3ProcgenAgent, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            num_actions=num_actions,
            use_sde=use_sde,
            arch=arch,
            base_kwargs=base_kwargs,
            **kwargs
        )

    def get_value(self, x):
        new_x = x.permute((0, 3, 1, 2)) / 255.0
        value, _ = self.network(new_x)
        return value
    
    def get_action_and_value(self, x, action=None, full_log_probs=False, deterministic=False):
        new_x = x.permute((0, 3, 1, 2)) / 255.0
        value, actor_features = self.network(new_x)
        dist = self.dist(actor_features)

        if action is None:
            action = dist.mode() if deterministic else dist.sample()
        action_log_probs = torch.squeeze(dist.log_probs(action))
        dist_entropy = dist.entropy()

        if full_log_probs:
            log_probs = torch.log(dist.probs)
            return torch.squeeze(action), action_log_probs, dist_entropy, value, log_probs

        return torch.squeeze(action), action_log_probs, dist_entropy, value