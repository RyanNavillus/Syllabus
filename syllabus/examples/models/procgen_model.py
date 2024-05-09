from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))


init_relu_ = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu')
)


init_tanh_ = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
)


def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    """
    Flatten a tensor
    """
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class FixedCategorical(torch.distributions.Categorical):
    """
    Categorical distribution object
    """
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    """
    Categorical distribution (NN module)
    """
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class Policy(nn.Module):
    """
    Actor-Critic module
    """
    def __init__(self, obs_shape, num_actions, arch='small', base_kwargs=None):
        super(Policy, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            if arch == 'small':
                base = SmallNetBase
            else:
                base = ResNetBase
        elif len(obs_shape) == 1:
            base = MLPBase

        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = Categorical(self.base.output_size, num_actions)
        self.critic_linear = init_(nn.Linear(base_kwargs.get("hidden_size"), 1))

        self.latent_dim_vf = 256
        self.latent_dim_pi = 256

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward_critic(self, features):
        values, _ = self.base(features)
        return values

    def forward_actor(self, features):
        value, actor_features = self.base(features)
        dist = self.dist(actor_features)
        dist = dist.sample().float()
        return dist

    def forward(self, inputs):
        value, actor_features, rnn_hxs = self.base(inputs, None, None)
        dist = self.dist(actor_features)
        return dist.sample(), value

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_dist = dist.logits

        return value, action, action_log_dist

    def get_value(self, inputs):
        value, _, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        return 1

    @property
    def output_size(self):
        return self._hidden_size


class MLPBase(NNBase):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        self.actor = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.train()

    def forward(self, inputs):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return hidden_critic, hidden_actor


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ResNetBase(NNBase):
    """
    Residual Network
    """
    def __init__(self, num_inputs=16, recurrent=False, hidden_size=256, channels=[16, 32, 32]):
        super(ResNetBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))
        return x


class SmallNetBase(NNBase):
    """
    Residual Network
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=256):
        super(SmallNetBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.conv1 = Conv2d_tf(3, 16, kernel_size=8, stride=4)
        self.conv2 = Conv2d_tf(16, 32, kernel_size=4, stride=2)

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))

        apply_init_(self.modules())

        self.train()

    def forward(self, inputs):
        x = inputs

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        return x


class ProcgenAgent(Policy):
    def __init__(self, obs_shape, num_actions, arch='small', base_kwargs=None):

        h, w, c = obs_shape
        shape = (c, h, w)
        super().__init__(shape, num_actions, arch=arch, base_kwargs=base_kwargs)

    def get_value(self, x):
        new_x = x.permute((0, 3, 1, 2)) / 255.0
        value, _ = self.base(new_x)
        return value

    def get_action_and_value(self, x, action=None, full_log_probs=False, deterministic=False):
        new_x = x.permute((0, 3, 1, 2)) / 255.0
        value, actor_features = self.base(new_x)
        dist = self.dist(actor_features)

        if action is None:
            action = dist.mode() if deterministic else dist.sample()
        action_log_probs = torch.squeeze(dist.log_probs(action))
        dist_entropy = dist.entropy()

        if full_log_probs:
            log_probs = torch.log(dist.probs)
            return torch.squeeze(action), action_log_probs, dist_entropy, value, log_probs

        return torch.squeeze(action), action_log_probs, dist_entropy, value

class SB3ResNetBase(ResNetBase):
    def __init__(self, observation_space, features_dim: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, inputs):
        return super().forward(inputs / 255.0)

class Sb3ProcgenAgent(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        hidden_size=256,
        **kwargs
    ):

        self.shape = observation_space.shape
        self.num_actions = action_space.n
        self.hidden_size = hidden_size

        super(Sb3ProcgenAgent, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        self.ortho_init = False
        self.apply(self.init_weights)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(self.hidden_size, [], None)

    def init_weights(self, m, **kwargs):
        if m is self.action_net:
            nn.init.orthogonal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)
        elif m is self.value_net:
            nn.init.orthogonal_(m.weight, gain=1.0) 
            nn.init.constant_(m.bias, 0)


        
        