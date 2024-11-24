import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))


def init_relu_(m): return init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu')
)


def init_tanh_(m): return init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)


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

    def _compute_padding(self, inputs, dim):
        input_size = inputs.size(dim + 2)
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

        def init_(m):
            return init(
                m,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
                gain=0.01
            )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """

    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """

    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
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

    def __init__(self, num_inputs, hidden_size=256, channels=[16, 32, 32]):
        super(ResNetBase, self).__init__(hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.flatten = Flatten()
        self.relu = nn.ReLU()

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


class ProcgenAgent(nn.Module):
    def __init__(self, obs_shape, num_actions, arch='small', base_kwargs=None):
        super(ProcgenAgent, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        self.base = ResNetBase(obs_shape[2], **base_kwargs)
        self.critic = init_(nn.Linear(256, 1))
        self.actor = Categorical(256, num_actions)

        apply_init_(self.modules())

    def get_value(self, inputs):
        new_inputs = inputs.permute((0, 3, 1, 2)) / 255.0
        hidden = self.base(new_inputs)
        return self.critic(hidden)

    def get_action(self, inputs):
        new_inputs = inputs.permute((0, 3, 1, 2)) / 255.0
        hidden = self.base(new_inputs)
        dist = self.actor(hidden)
        action = dist.sample()
        return torch.squeeze(action)

    def get_action_and_value(self, inputs, action=None, deterministic=False):
        new_inputs = inputs.permute((0, 3, 1, 2)) / 255.0
        hidden = self.base(new_inputs)
        value = self.critic(hidden)
        dist = self.actor(hidden)

        if action is None:
            action = dist.mode() if deterministic else dist.sample()
        action_log_probs = torch.squeeze(dist.log_probs(action))
        dist_entropy = dist.entropy()

        return torch.squeeze(action), action_log_probs, dist_entropy, value


class ProcgenLSTMAgent(nn.Module):
    def __init__(self, obs_shape, num_actions, base_kwargs=None):
        super(ProcgenLSTMAgent, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        hidden_size = base_kwargs.get("hidden_size", 256)

        self.base = ResNetBase(obs_shape[2], **base_kwargs)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.critic = init_(nn.Linear(hidden_size, 1))
        self.actor = layer_init(nn.Linear(hidden_size, num_actions), std=0.01)

        apply_init_(self.modules())

    def get_states(self, inputs, lstm_state, done):
        new_inputs = inputs.permute((0, 3, 1, 2)) / 255.0
        hidden = self.base(new_inputs)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, inputs, lstm_state, done):
        hidden, lstm_state = self.get_states(inputs, lstm_state, done)
        return self.critic(hidden), lstm_state

    def get_action(self, inputs, lstm_state, done):
        hidden, lstm_state = self.get_states(inputs, lstm_state, done)
        dist = torch.distributions.categorical.Categorical(logits=self.actor(hidden))
        action = dist.sample()
        return torch.squeeze(action), lstm_state

    def get_action_and_value(self, inputs, lstm_state, done, action=None, deterministic=False):
        hidden, lstm_state = self.get_states(inputs, lstm_state, done)

        value = self.critic(hidden)
        dist = torch.distributions.categorical.Categorical(logits=self.actor(hidden))

        if action is None:
            action = dist.mode() if deterministic else dist.sample()
        action_log_probs = torch.squeeze(dist.log_prob(action))
        dist_entropy = dist.entropy()

        return torch.squeeze(action), action_log_probs, dist_entropy, value, lstm_state
