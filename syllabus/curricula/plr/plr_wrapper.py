import torch
import numpy as np

from syllabus.curricula import LevelSampler
from syllabus.core import Curriculum
from typing import Any, Callable, Dict, List, Union, Tuple


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, action_space, requires_value_buffers):
        self._requires_value_buffers = requires_value_buffers
        self.action_log_dist = torch.zeros(num_steps, num_processes, action_space.n)
        self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        if requires_value_buffers:
            self.returns = torch.zeros(num_steps + 1, num_processes, 1)
            self.rewards = torch.zeros(num_steps, num_processes, 1)
            self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.action_log_dist = self.action_log_dist.to(device)
        self.masks = self.masks.to(device)
        self.level_seeds = self.level_seeds.to(device)
        if self._requires_value_buffers:
            self.rewards = self.rewards.to(device)
            self.value_preds = self.value_preds.to(device)
            self.returns = self.returns.to(device)

    def insert(self, action_log_dist, masks, value_preds=None, rewards=None, level_seeds=None):
        if self._requires_value_buffers:
            assert value_preds is not None and rewards is not None, f"Selected strategy {self._requires_value_buffers} requires value_preds and rewards"
            if len(rewards.shape) == 3:
                rewards = rewards.squeeze(2)
            self.value_preds[self.step].copy_(torch.as_tensor(value_preds))
            self.rewards[self.step].copy_(torch.as_tensor(rewards)[:, None])
            self.masks[self.step + 1].copy_(torch.as_tensor(masks)[:, None])

        self.action_log_dist[self.step].copy_(action_log_dist)
        if level_seeds is not None:
            self.level_seeds[self.step].copy_(torch.as_tensor(level_seeds)[:, None])
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self,
                        next_value,
                        gamma,
                        gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]


class PrioritizedLevelReplay(Curriculum):
    def __init__(self,
                 level_sampler_args,
                 level_sampler_kwargs,
                 action_space,
                 *curriculum_args,
                 device="cuda",
                 num_steps=256,
                 num_processes=64,
                 gamma=0.999,
                 gae_lambda=0.95,
                 **curriculum_kwargs):
        self._strategy = level_sampler_kwargs.get("strategy", "random")
        if "num_actors" in level_sampler_kwargs:
            print(f"Overwriting 'num_actors' {level_sampler_kwargs['num_actors']} in level sampler kwargs with PLR num_processes {num_processes}.")
        level_sampler_kwargs["num_actors"] = num_processes
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self._num_steps = num_steps    # Number of steps stored in rollouts and used to update level sampler
        self._num_processes = num_processes      # Number of parallel environments
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._level_sampler = LevelSampler(*level_sampler_args, **level_sampler_kwargs)
        self._rollouts = RolloutStorage(self._num_steps, self._num_processes, action_space, self._level_sampler.requires_value_buffers)
        self._rollouts.to(device)

    def _on_demand(self, metrics: Dict):
        """
        Update the curriculum with arbitrary inputs.
        """
        action_log_dist = metrics["action_log_dist"]
        masks = metrics["masks"]
        level_seeds = metrics["level_seeds"]
        value = next_value = rew = None
        if self._level_sampler.requires_value_buffers:
            if "value" not in metrics or "rew" not in metrics:
                raise KeyError(f"'value' and 'rew' must be provided in every update for the strategy {self._strategy}.")
            value = metrics["value"]
            rew = metrics["rew"]

        # Update rollouts
        self._rollouts.insert(action_log_dist, masks, value_preds=value, rewards=rew, level_seeds=level_seeds)

        # Update level sampler
        if self._rollouts.step == self._num_steps - 1:
            if self._level_sampler.requires_value_buffers:
                if "next_value" not in metrics:
                    raise KeyError("'next_value' must be provided in the update every {self.num_steps} steps for the strategy {self._strategy}.")
                next_value = metrics["next_value"]
                self._rollouts.compute_returns(next_value, self._gamma, self._gae_lambda)
            self._level_sampler.update_with_rollouts(self._rollouts)
            self._rollouts.after_update()
            self._level_sampler.after_update()

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        return self._level_sampler.sample_weights()

    def sample(self, k: int = 1) -> Union[List, Any]:
        sample = [self._level_sampler.sample() for k in range(k+1)]
        return sample
