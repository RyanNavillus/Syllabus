from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from .procgen_model import Policy

class Sb3ProcgenAgent(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        arch='small', base_kwargs=None, **kwargs
    ):
        self.ortho_init = False

        self.shape = observation_space.shape
        self.num_actions = action_space.n
        self.arch=arch
        self.base_kwargs=base_kwargs 

        super(Sb3ProcgenAgent, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            **kwargs,
        )
        
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Policy(self.shape, self.num_actions, self.arch, self.base_kwargs)