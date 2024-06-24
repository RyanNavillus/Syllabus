import gymnasium as gym
import torch

class RolloutStorage(object):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        requires_value_buffers: bool,
        action_space: gym.Space = None,
    ):
        self._requires_value_buffers = requires_value_buffers
        self.tasks = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        if requires_value_buffers:
            self.returns = torch.zeros(num_steps + 1, num_processes, 1)
            self.rewards = torch.zeros(num_steps, num_processes, 1)
            self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        else:
            if action_space is None:
                raise ValueError(
                    "Action space must be provided to PLR for strategies 'policy_entropy', 'least_confidence', 'min_margin'"
                )
            self.action_log_dist = torch.zeros(num_steps, num_processes, action_space.n)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.masks = self.masks.to(device)
        self.tasks = self.tasks.to(device)
        if self._requires_value_buffers:
            self.rewards = self.rewards.to(device)
            self.value_preds = self.value_preds.to(device)
            self.returns = self.returns.to(device)
        else:
            self.action_log_dist = self.action_log_dist.to(device)

    def insert(self, masks, action_log_dist=None, value_preds=None, rewards=None, tasks=None):
        if self._requires_value_buffers:
            assert (value_preds is not None and rewards is not None), "Selected strategy requires value_preds and rewards"
            if len(rewards.shape) == 3:
                rewards = rewards.squeeze(2)
            self.value_preds[self.step].copy_(torch.as_tensor(value_preds))
            self.rewards[self.step].copy_(torch.as_tensor(rewards))
            self.masks[self.step + 1].copy_(torch.as_tensor(masks))
        else:
            self.action_log_dist[self.step].copy_(action_log_dist)
        if tasks is not None:
            # assert isinstance(tasks[0], (int, torch.int32)), "Provided task must be an integer"
            self.tasks[self.step].copy_(torch.as_tensor(tasks))
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        assert self._requires_value_buffers, "Selected strategy does not use compute_rewards."
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step]
                + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]