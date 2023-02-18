import gym
import numpy as np
import typing
from typing import Any, List, Union
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from itertools import product


class Curriculum:
    """
    Base class and API for defining curricula to interface with Gym environments.
    """
    def __init__(self, task_space: gym.Space, random_start_tasks: int = 0, wandb_run=None) -> None:
        self.task_space = task_space
        self.wandb_run = wandb_run
        self.random_start_tasks = random_start_tasks
        self.completed_tasks = 0

    def _sum_axes(list_or_size: Union[np.ndarray, int]):
        if isinstance(list_or_size, int):
            return list_or_size
        elif isinstance(list_or_size, list):
            return sum([Curriculum._sum_axes(x) for x in list_or_size])

    def _enumerate_axes(list_or_size: Union[np.ndarray, int]):
        if isinstance(list_or_size, int):
            return tuple(range(list_or_size))
        elif isinstance(list_or_size, list):
            return tuple(product(*[Curriculum._enumerate_axes(x) for x in list_or_size]))

    @property
    def _n_tasks(self, task_space: gym.Space = None) -> int:
        """
        Return the number of discrete tasks in the task_space.
        Returns None for continuous spaces.
        Graph space not implemented.
        """
        # TODO: Test these implementations
        if task_space is None:
            task_space = self.task_space

        if isinstance(task_space, Discrete):
            return task_space.n
        elif isinstance(task_space, Box):
            return None
        elif isinstance(task_space, gym.spaces.Tuple):
            return sum([self._n_tasks(s) for s in task_space.spaces])
        elif isinstance(task_space, Sequence):
            return sum([self._n_tasks(s) for s in task_space.spaces])
        elif isinstance(task_space, Dict):
            return sum([self._n_tasks(s) for s in task_space.spaces.values()])
        elif isinstance(task_space, MultiBinary):
            return Curriculum._sum_axes(task_space.nvec)
        elif isinstance(task_space, MultiDiscrete):
            return Curriculum._sum_axes(task_space.nvec)
        elif isinstance(task_space, Text):
            return None
        else:
            raise NotImplementedError

    @property
    def _tasks(self, task_space: gym.Space = None, sample_interval: float = None) -> List[tuple]:
        """
        Return the full list of discrete tasks in the task_space.
        Return a sample of the tasks for continuous spaces if sample_interval is specified.
        Can be overridden to exclude invalid tasks within the space.
        """
        if task_space is None:
            task_space = self.task_space

        if isinstance(task_space, Discrete):
            return list(range(task_space.n))
        elif isinstance(task_space, Box):
            raise NotImplementedError
        elif isinstance(task_space, gym.spaces.Tuple):
            raise NotImplementedError
        elif isinstance(task_space, Dict):
            raise NotImplementedError
        elif isinstance(task_space, MultiBinary):
            return list(Curriculum._enumerate_axes(task_space.nvec))
        elif isinstance(task_space, MultiDiscrete):
            return list(Curriculum._enumerate_axes(task_space.nvec))
        else:
            raise NotImplementedError

    def complete_task(self, task: typing.Any, success_prob: float) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        self.completed_tasks += 1

    def on_step(self, obs, rew, done, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        raise NotImplementedError

    def on_step_batch(self, step_results: List[typing.Tuple[int, int, int, int]]) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        for step_result in step_results:
            self.on_step(*step_result)

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        return [1.0 / self._n_tasks for _ in range(self._n_tasks)]

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        if self.random_start_tasks > 0 and self.completed_tasks < self.random_start_tasks:
            task_dist = [0.0 / self._n_tasks for _ in range(self._n_tasks)]
            task_dist[0] = 1.0
        else:
            task_dist = self._sample_distribution()
        self.log_task_dist(task_dist)
        return np.random.choice(self._tasks, size=k, p=task_dist)

    def log_task_dist(self, task_dist: List[float], check_dist=True):
        """
        Log the task distribution to wandb.

        Paramaters:
            task_dist: List of task probabilities. Must be a valid probability distribution.
        """
        if check_dist:
            assert sum(task_dist) - 1.0 < 0.0001, "Task distribution must be a valid probability distribution."
        print(task_dist)
        if self.wandb_run:
            self.wandb_run.log({"task_dist": self.task_space.sample()}, commit=False)
