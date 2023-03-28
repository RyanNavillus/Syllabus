import typing
from typing import Any, Callable, List, Tuple, Union

import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

import wandb
from syllabus.core import enumerate_axes


class Curriculum:
    """
    Base class and API for defining curricula to interface with Gym environments.
    """
    def __init__(self, task_space: gym.Space, random_start_tasks: int = 0, task_names: Callable = None) -> None:
        self.task_space = task_space
        self.random_start_tasks = random_start_tasks
        self.completed_tasks = 0
        self.task_names = task_names
        self.n_updates = 0

    def _sum_axes(list_or_size: Union[list, int]):
        if isinstance(list_or_size, int) or isinstance(list_or_size, np.int64):
            return list_or_size
        elif isinstance(list_or_size, list) or isinstance(list_or_size, np.ndarray):
            return np.prod([Curriculum._sum_axes(x) for x in list_or_size])
        else:
            raise NotImplementedError(f"{type(list_or_size)}")

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
        elif isinstance(task_space, Dict):
            return sum([self._n_tasks(s) for s in task_space.spaces.values()])
        elif isinstance(task_space, MultiBinary):
            return Curriculum._sum_axes(task_space.nvec)
        elif isinstance(task_space, MultiDiscrete):
            return Curriculum._sum_axes(task_space.nvec)
        else:
            raise NotImplementedError(f"Unsupported task space type: {type(task_space)}")

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
            return list(enumerate_axes(task_space.nvec))
        elif isinstance(task_space, MultiDiscrete):
            return list(enumerate_axes(task_space.nvec))
        else:
            raise NotImplementedError

    def _complete_task(self, task: typing.Any, success_prob: Tuple[float, bool]) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        self.completed_tasks += 1

    def _on_step(self, obs, rew, done, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        raise NotImplementedError("Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def _on_step_batch(self, step_results: List[typing.Tuple[int, int, int, int]]) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        for step_result in step_results:
            self._on_step(*step_result)

    def _on_episode(self, episode_return: float, trajectory: List = None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        raise NotImplementedError("Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def _on_demand(self, metrics: Dict):
        """
        Update the curriculum with arbitrary inputs.
        """
        raise NotImplementedError

    def update_curriculum(self, update_data: Dict):
        """
        Update the curriculum with the specified update type.
        """
        update_type = update_data["update_type"]
        args = update_data["metrics"]

        if update_type == "step":
            self._on_step(*args)
        elif update_type == "step_batch":
            self._on_step_batch(*args)
        elif update_type == "episode":
            self._on_episode(*args)
        elif update_type == "on_demand":
            # Directly pass metrics without expanding
            self._on_demand(args)
        elif update_type == "complete":
            self._complete_task(*args)
        elif update_type == "noop":
            # Used to request tasks from the synchronization layer
            pass
        else:
            raise NotImplementedError(f"Update type {update_type} not implemented.")
        self.n_updates += 1

    def batch_update_curriculum(self, update_data: List[Dict]):
        """
        Update the curriculum with the specified update type.
        """
        for update in update_data:
            self.update_curriculum(update)

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

        # Use list of indices because np.choice does not play nice with tuple tasks
        tasks = self._tasks
        n_tasks = len(tasks)
        task_idx = np.random.choice(list(range(n_tasks)), size=k, p=task_dist)
        return [tasks[i] for i in task_idx]

    def log_metrics(self, step=None):
        """
        Log the task distribution to wandb.

        Paramaters:
            task_dist: List of task probabilities. Must be a valid probability distribution.
        """
        try:
            task_dist = self._sample_distribution()
            if self.task_names:
                dist_dict = {f"{self.task_names[idx]}_prob": prob for idx, prob in enumerate(task_dist)}
            else:
                dist_dict = {f"task_{idx}_prob": prob for idx, prob in enumerate(task_dist)}
            wandb.log({"curriculum": dist_dict}, step=step)
        except wandb.errors.Error:
            # No need to crash over logging :)
            pass
