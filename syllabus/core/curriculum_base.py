import typing
import itertools
from typing import Any, Callable, List, Tuple, Union
import time 
import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

import wandb
from syllabus.core import enumerate_axes
from syllabus.task_space import TaskSpace

# TODO: Move non-generic logic to Uniform class. Allow subclasses to call super for generic error handling
class Curriculum:
    """
    Base class and API for defining curricula to interface with Gym environments.
    """

    def __init__(self, task_space: TaskSpace, random_start_tasks: int = 0, task_names: Callable = None) -> None:
        assert isinstance(task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        self.task_space = task_space
        self.random_start_tasks = random_start_tasks
        self.completed_tasks = 0
        self.task_names = task_names
        self.n_updates = 0

        if self.num_tasks == 0:
            print("Warning: Task space is empty. This will cause errors during sampling if no tasks are added.")

    @property
    def num_tasks(self) -> int:
        # TODO: Cache results
        return self.task_space.num_tasks

    @property
    def tasks(self) -> List[tuple]:
        return self.task_space.tasks
        
    def add_task(self, task: tuple) -> None:
        """
        Add a task to the curriculum.
        """
        raise NotImplementedError("This curriculum does not support adding tasks after initialization.")

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
        raise NotImplementedError("This curriculum does not require step updates. Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

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

    # TODO: Move to curriculum sync wrapper?
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
        elif update_type == "add_task":
            self.add_task(args)
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
        raise NotImplementedError

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        assert self.num_tasks > 0, "Task space is empty. Please add tasks to the curriculum before sampling."

        if self.random_start_tasks > 0 and self.completed_tasks < self.random_start_tasks:
            task_dist = [0.0 / self.num_tasks for _ in range(self.num_tasks)]
            task_dist[0] = 1.0
        else:
            task_dist = self._sample_distribution()

        # Use list of indices because np.choice does not play nice with tuple tasks
        tasks = self.tasks
        n_tasks = len(tasks)
        task_idx = np.random.choice(list(range(n_tasks)), size=k, p=task_dist)
        return [tasks[i] for i in task_idx]

    def log_metrics(self, writer, step=None):
        """
        Log the task distribution to wandb.

        Paramaters:
            task_dist: List of task probabilities. Must be a valid probability distribution.
        """
        try:
            task_dist = self._sample_distribution()
            if self.task_names:
                for idx, prob in enumerate(task_dist):
                    writer.add_scalar(f"curriculum/task_{self.task_space.task_name(idx)}_prob", prob, step)
            else:
                for idx, prob in enumerate(task_dist):
                    writer.add_scalar(f"curriculum/task_{idx}_prob", prob, step)
        except wandb.errors.Error:
            # No need to crash over logging :)
            pass

