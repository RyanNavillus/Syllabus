import gym
import torch
import numpy as np

from syllabus.curricula import TaskSampler
from syllabus.core import Curriculum, UsageError
from typing import Any, Callable, Dict, List, Union, Tuple


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, action_space, requires_value_buffers):
        self._requires_value_buffers = requires_value_buffers
        self.action_log_dist = torch.zeros(num_steps, num_processes, action_space.n)
        self.tasks = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
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
        self.tasks = self.tasks.to(device)
        if self._requires_value_buffers:
            self.rewards = self.rewards.to(device)
            self.value_preds = self.value_preds.to(device)
            self.returns = self.returns.to(device)

    def insert(self, action_log_dist, masks, value_preds=None, rewards=None, tasks=None):
        if self._requires_value_buffers:
            assert value_preds is not None and rewards is not None, f"Selected strategy {self._requires_value_buffers} requires value_preds and rewards"
            if len(rewards.shape) == 3:
                rewards = rewards.squeeze(2)
            self.value_preds[self.step].copy_(torch.as_tensor(value_preds))
            self.rewards[self.step].copy_(torch.as_tensor(rewards)[:, None])
            self.masks[self.step + 1].copy_(torch.as_tensor(masks)[:, None])

        self.action_log_dist[self.step].copy_(action_log_dist)
        if tasks is not None:
            self.tasks[self.step].copy_(torch.as_tensor(tasks)[:, None])
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
                 task_space,
                 task_sampler_kwargs,
                 action_space,
                 *curriculum_args,
                 device="cuda",
                 num_steps=256,
                 num_processes=64,
                 gamma=0.999,
                 gae_lambda=0.95,
                 **curriculum_kwargs):
        self._strategy = task_sampler_kwargs.get("strategy", "random")
        if not isinstance(task_space, gym.spaces.Discrete) and not isinstance(task_space, gym.spaces.MultiDiscrete):
            raise ValueError(f"Task space must be discrete or multi-discrete, got {task_space}.")

        if "num_actors" in task_sampler_kwargs:
            print(f"Overwriting 'num_actors' {task_sampler_kwargs['num_actors']} in task sampler kwargs with PLR num_processes {num_processes}.")
        task_sampler_kwargs["num_actors"] = num_processes
        super().__init__(task_space, *curriculum_args, **curriculum_kwargs)
        self._num_steps = num_steps    # Number of steps stored in rollouts and used to update task sampler
        self._num_processes = num_processes      # Number of parallel environments
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._task_sampler = TaskSampler(task_space, action_space, **task_sampler_kwargs)
        self._rollouts = RolloutStorage(self._num_steps, self._num_processes, action_space, self._task_sampler.requires_value_buffers)
        self._rollouts.to(device)
        self.num_updates = 0    # Used to ensure proper usage
        self.num_samples = 0    # Used to ensure proper usage

    def _on_demand(self, metrics: Dict):
        """
        Update the curriculum with arbitrary inputs.
        """
        self.num_updates += 1
        action_log_dist = metrics["action_log_dist"]
        masks = metrics["masks"]
        tasks = metrics["tasks"]
        value = next_value = rew = None
        if self._task_sampler.requires_value_buffers:
            if "value" not in metrics or "rew" not in metrics:
                raise KeyError(f"'value' and 'rew' must be provided in every update for the strategy {self._strategy}.")
            value = metrics["value"]
            rew = metrics["rew"]

        # Update rollouts
        self._rollouts.insert(action_log_dist, masks, value_preds=value, rewards=rew, tasks=tasks)

        # Update task sampler
        if self._rollouts.step == 0:
            if self._task_sampler.requires_value_buffers:
                if "next_value" not in metrics:
                    raise KeyError("'next_value' must be provided in the update every {self.num_steps} steps for the strategy {self._strategy}.")
                next_value = metrics["next_value"]
                self._rollouts.compute_returns(next_value, self._gamma, self._gae_lambda)
            self._task_sampler.update_with_rollouts(self._rollouts)
            self._rollouts.after_update()
            self._task_sampler.after_update()

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        return self._task_sampler.sample_weights()

    def sample(self, k: int = 1) -> Union[List, Any]:
        self.num_samples += 1
        return [self._task_sampler.sample() for k in range(k+1)]

    def _on_step(self, obs, rew, done, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        raise NotImplementedError("PrioritizedLevelReplay does not support the step updates. Use on_demand from the learner process.")

    def _on_step_batch(self, step_results: List[Tuple[int, int, int, int]]) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        raise NotImplementedError("PrioritizedLevelReplay does not support the step updates. Use on_demand from the learner process.")

    def _on_episode(self, episode_return: float, trajectory: List = None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        raise NotImplementedError("PrioritizedLevelReplay does not support the episode updates. Use on_demand from the learner process.")

    def _complete_task(self, task: Any, success_prob: float) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        if self.num_updates == 0 and self.num_samples > self._num_processes * 2:
            raise UsageError("PLR has not been updated yet. Please call update_curriculum() in your learner process.")
