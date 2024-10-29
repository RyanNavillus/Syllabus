import dataclasses
import math
import os
import json
import warnings
import numpy as np
import torch
from syllabus.task_space import TaskSpace
from gymnasium.spaces import Discrete
from collections import deque, defaultdict


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

        self.tasks = self.task_space.get_tasks()
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

    def log_metrics(self, writer, step=None, log_full_dist=False):
        """Log the statistics of the first 5 tasks to the provided tensorboard writer.

        :param writer: Tensorboard summary writer.
        """
        try:
            import wandb
            tasks_to_log = self.tasks
            if len(self.tasks) > 10 and not log_full_dist:
                warnings.warn("Only logging stats for 5 tasks.")
                tasks_to_log = self.tasks[:10]
            logs = []
            for idx in tasks_to_log:
                if self.episode_returns[idx].n > 0:
                    name = self.task_names(list(self.task_space.tasks)[idx], idx)
                    logs.append((f"tasks/{name}_episode_return", self.episode_returns[idx].mean(), step))
                    logs.append((f"tasks/{name}_episode_length", self.episode_lengths[idx].mean(), step))
            for name, prob, step in logs:
                if writer == wandb:
                    writer.log({name: prob}, step=step)
                else:
                    writer.add_scalar(name, prob, step)
        except ImportError:
            warnings.warn("Wandb is not installed. Skipping logging.")
        except wandb.errors.Error:
            # No need to crash over logging :)
            warnings.warn("Failed to log curriculum stats to wandb.")

    def normalize(self, reward, task):
        """
        Normalize reward by task.
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
        """
        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()  # Use .item() to convert numpy types to native Python types
            raise TypeError
        stats = json.dumps(self.stats, default=convert_numpy)
        with open(os.path.join(output_path, 'task_specific_stats.json'), "w") as file:
            file.write(stats)
