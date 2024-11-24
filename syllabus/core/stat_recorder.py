import dataclasses
import json
import math
import os
import warnings
from collections import deque

import numpy as np
import torch
from gymnasium.spaces import Discrete

from syllabus.task_space import TaskSpace


@dataclasses.dataclass
class StatMean:
    # Compute using Welford'd Online Algorithm
    # Algo: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Math: https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
    n: int = 0
    mu: float = 0
    m2: float = 0

    def result(self):
        if self.n == 0:
            return None
        return self.mu

    def mean(self):
        return self.mu

    def std(self):
        if self.n < 1:
            return None
        return math.sqrt(self.m2 / self.n)

    def __sub__(self, other):
        assert isinstance(other, StatMean)
        n_new = self.n - other.n
        if n_new == 0:
            return StatMean(0, 0, 0)
        mu_new = (self.mu * self.n - other.mu * other.n) / n_new
        delta = other.mu - mu_new
        m2_new = self.m2 - other.m2 - (delta**2) * n_new * other.n / self.n
        return StatMean(n_new, mu_new, m2_new)

    def __iadd__(self, other):
        if isinstance(other, StatMean):
            other_n = other.n
            other_mu = other.mu
            other_m2 = other.m2
        elif isinstance(other, torch.Tensor):
            other_n = other.numel()
            other_mu = other.mean().item()
            other_m2 = ((other - other_mu) ** 2).sum().item()
        else:
            other_n = 1
            other_mu = other
            other_m2 = 0
        # See parallelized Welford in wiki
        new_n = other_n + self.n
        delta = other_mu - self.mu
        self.mu += delta * (other_n / max(new_n, 1))
        delta2 = other_mu - self.mu
        self.m2 += other_m2 + (delta2**2) * (self.n * other_n / max(new_n, 1))
        self.n = new_n
        return self

    def reset(self):
        self.mu = 0
        self.n = 0

    def __repr__(self):
        return repr(self.result())


class StatRecorder:
    """
    Individual statistics tracking for each task.
    """

    def __init__(self, task_space: TaskSpace, calc_past_n=None, task_names=None):
        """Initialize the StatRecorder"""

        self.task_space = task_space
        self.calc_past_n = calc_past_n
        self.task_names = task_names if task_names is not None else lambda task, idx: idx

        assert isinstance(
            self.task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        assert isinstance(self.task_space.gym_space,
                          Discrete), f"Only Discrete task spaces are supported. Got {type(task_space.gym_space)}"

        self.tasks = self.task_space.tasks
        self.num_tasks = self.task_space.num_tasks

        self.episode_returns = {task: StatMean() for task in self.tasks}
        self.episode_lengths = {task: StatMean() for task in self.tasks}

    def record(self, episode_return: float, episode_length: int, episode_task, env_id=None):
        """
        Record the length and return of an episode for a given task.

        :param episode_length: Length of the episode, i.e. the total number of steps taken during the episode
        :param episodic_return: Total return for the episode
        :param episode_task: Identifier for the task
        """
        if episode_task not in self.tasks:
            raise ValueError(f"Stat recorder received unknown task {episode_task}.")

        self.episode_returns[episode_task] += episode_return
        self.episode_lengths[episode_task] += episode_length

    def get_metrics(self, log_n_tasks=1):
        """Log the statistics of the first 5 tasks to the provided tensorboard writer.

        :param writer: Tensorboard summary writer.
        :param log_n_tasks: Number of tasks to log statistics for. Use -1 to log all tasks.
        """
        tasks_to_log = self.tasks
        if len(self.tasks) > log_n_tasks and log_n_tasks != -1:
            warnings.warn(f"Too many tasks to log {len(self.tasks)}. Only logging stats for 1 task.", stacklevel=2)
            tasks_to_log = self.tasks[:log_n_tasks]

        logs = []
        for idx in tasks_to_log:
            if self.episode_returns[idx].n > 0:
                name = self.task_names(list(self.task_space.tasks)[idx], idx)
                logs.append((f"tasks/{name}_episode_return", self.episode_returns[idx].mean()))
                logs.append((f"tasks/{name}_episode_length", self.episode_lengths[idx].mean()))
        return logs

    def normalize(self, reward, task):
        """
        Normalize reward by task.

        :param reward: Reward to normalize
        :param task: Task to normalize reward by
        """
        task_return_stats = self.episode_returns[task]
        reward_std = task_return_stats.std()
        normalized_reward = deque(maxlen=reward.maxlen)
        for r in reward:
            normalized_reward.append(r / max(0.01, reward_std))
        return normalized_reward

    def save_statistics(self, output_path):
        """
        Write task-specific statistics to file.

        :param output_path: Path to save the statistics file.
        """
        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()  # Use .item() to convert numpy types to native Python types
            raise TypeError
        stats = json.dumps(self.episode_returns, default=convert_numpy)
        with open(os.path.join(output_path, 'task_specific_stats.json'), "w", encoding="utf-8") as file:
            file.write(stats)
