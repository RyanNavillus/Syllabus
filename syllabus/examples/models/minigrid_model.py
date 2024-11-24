import numpy as np
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_(m):
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))


def init_relu_(m):
    return init(
        m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu")
    )


def init_tanh_(m):
    return init(
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

        self.linear = init(
            nn.Linear(num_inputs, num_outputs),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01
        )

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class MinigridPolicy(nn.Module):
    """
    Actor-Critic module
    """

    def __init__(self, obs_shape, num_actions, arch="small", base_kwargs=None):
        super(MinigridPolicy, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        final_channels = 32 if arch == "small" else 64

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, final_channels, (2, 2)),
            nn.ReLU(),
        )
        n = obs_shape[-2]
        m = obs_shape[-1]
        self.image_embedding_size = (
            ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * final_channels
        )
        self.embedding_size = self.image_embedding_size

        # Define actor's model
        self.actor_base = nn.Sequential(
            init_tanh_(nn.Linear(self.embedding_size, 64)),
            nn.Tanh(),
        )

        # Define critic's model
        self.critic = nn.Sequential(
            init_tanh_(nn.Linear(self.embedding_size, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 1)),
        )

        self.dist = Categorical(64, num_actions)

        apply_init_(self.modules())

        self.train()

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1

    def forward(self):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        x = inputs
        x = self.image_conv(x)
        x = x.flatten(1, -1)
        actor_features = self.actor_base(x)
        value = self.critic(x)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # action_log_probs = dist.log_probs(action)
        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_dist, dist_entropy

    def get_value(self, inputs):
        x = inputs
        x = self.image_conv(x)
        x = x.flatten(1, -1)
        return self.critic(x)

    def evaluate_actions(self, inputs, action):
        x = inputs
        x = self.image_conv(x)
        x = x.flatten(1, -1)
        actor_features = self.actor_base(x)
        value = self.critic(x)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class MinigridAgent(MinigridPolicy):
    def get_value(self, x):
        x = self.image_conv(x)
        x = x.flatten(1, -1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None, full_log_probs=False):
        x = self.image_conv(x)
        x = x.flatten(1, -1)
        actor_features = self.actor_base(x)
        value = self.critic(x)
        dist = self.dist(actor_features)

        action = torch.squeeze(dist.sample())

        action_log_probs = torch.squeeze(dist.log_probs(action))
        dist_entropy = dist.entropy()

        if full_log_probs:
            log_probs = torch.log(dist.probs)
            return action, action_log_probs, dist_entropy, value, log_probs

        return action, action_log_probs, dist_entropy, value
