import torch
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from syllabus.examples.models import SmallNetBase, ResNetBase, MLPBase, Categorical

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, arch='small', base_kwargs=None):
        super(CustomFeaturesExtractor, self).__init__(observation_space)

        if base_kwargs is None:
            base_kwargs = {}
        
        if len(observation_space.shape) == 3:
            if arch == 'small':
                network = SmallNetBase
            else:
                network = ResNetBase
        elif len(observation_space.shape) == 1:
                network = MLPBase

        self.network = network(observation_space.shape[0], **base_kwargs)
    
    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.network.recurrent_hidden_state_size
    
    def forward(self, observations, rnn_hxs = None, masks = None):
        return self.network(observations, rnn_hxs, masks)
    

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, num_actions, features_extractor_class=CustomFeaturesExtractor, arch='small', base_kwargs=None, **kwargs):
        super(CustomPolicy, self).__init__(
            observation_space, action_space, lr_schedule,
            features_extractor=features_extractor_class(observation_space, arch, base_kwargs),
            **kwargs
        )
        self.dist = Categorical(self.network.output_size, num_actions)
        self.latent_dim_pi = 256
        self.latent_dim_vf = 256

    def forward(self, input):
        value, actor_features, rnn_hxs = self.features_extractor(input, None, None)
        dist = self.dist(actor_features)
        return dist.sample(), value

    def predict(self, input, deterministic=False):
        value, actor_features = self.features_extractor.network(input)
        dist = self.dist(actor_features)
        action = dist.mode() if deterministic else dist.sample()
        action_log_dist = dist.logits
        return value, action, action_log_dist
    
    def get_value(self, input):
        value, _, _ = self.features_extractor(input)
        return value
    
    def evaluate_actions(self, input, rnn_hxs, masks, action):
        value, actor_features = self.features_extractor(input, rnn_hxs, masks)
        dist = self.dist(actor_features)
        
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hxs
    
class Sb3ProcgenAgent(CustomPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, num_actions, arch='small', base_kwargs=None):
        h, w, c = observation_space
        shape = (c, h, w)
        super(Sb3ProcgenAgent, self).__init__(
            shape, action_space, lr_schedule, num_actions,
            features_extractor_class=CustomFeaturesExtractor, arch=arch, base_kwargs=base_kwargs
        )

    def get_value(self, x):
        new_x = x.permute((0, 3, 1, 2)) / 255.0
        value, _ = self.features_extractor(new_x)
        return value
    
    def get_action_and_value(self, x, action=None, full_log_probs=False, deterministic=False):
        new_x = x.permute((0, 3, 1, 2)) / 255.0
        value, actor_features = self.features_extractor(new_x)
        dist = self.dist(actor_features)

        if action is None:
            action = dist.mode() if deterministic else dist.sample()
        action_log_probs = torch.squeeze(dist.log_probs(action))
        dist_entropy = dist.entropy()

        if full_log_probs:
            log_probs = torch.log(dist.probs)
            return torch.squeeze(action), action_log_probs, dist_entropy, value, log_probs

        return torch.squeeze(action), action_log_probs, dist_entropy, value