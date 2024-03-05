import typing
import warnings
from typing import Any, Callable, List, Tuple, Union

import numpy as np
from gymnasium.spaces import Dict
from syllabus.task_space import TaskSpace
import random


# TODO: Move non-generic logic to Uniform class. Allow subclasses to call super for generic error handling
class Curriculum:
    """Base class and API for defining curricula to interface with Gym environments.
    """

    def __init__(self, task_space: TaskSpace, task_names: Callable = None, warmup_strategy: str = None, warmup_samples: int = 0) -> None:
        """Initialize the base Curriculum

        :param task_space: the environment's task space from which new tasks are sampled
        TODO: Implement this in a way that works with any curriculum, maybe as a wrapper
        TODO: Use task space for this
        :param task_names: Names of the tasks in the task space, defaults to None
        """
        assert isinstance(task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        self.task_space = task_space
        self.completed_tasks = 0
        self.task_names = task_names
        self.n_updates = 0
        self.sampled_tasks = 0
        self.warmup_strategy = warmup_strategy
        self.warmup_tasks = warmup_samples

        if self.num_tasks == 0:
            warnings.warn("Task space is empty. This will cause errors during sampling if no tasks are added.")

    @property
    def num_tasks(self) -> int:
        """Counts the number of tasks in the task space.

        :return: Returns the number of tasks in the task space if it is countable, TODO: -1 otherwise
        """
        return self.task_space.num_tasks

    @property
    def tasks(self) -> List[tuple]:
        """List all of the tasks in the task space.

        :return: List of tasks if task space is enumerable, TODO: empty list otherwise?
        """
        return list(self.task_space.tasks)

    def add_task(self, task: typing.Any) -> None:
        # TODO
        raise NotImplementedError("This curriculum does not support adding tasks after initialization.")

    def update_task_progress(self, task: typing.Any, progress: Tuple[float, bool]) -> None:
        """Update the curriculum with a task and its progress.

        :param task: Task for which progress is being updated.
        :param progress: Progress toward completion or success rate of the given task. 1.0 or True typically indicates a complete task.
        """
        self.completed_tasks += 1

    def update_on_step(self, obs: typing.Any, rew: float, term: bool, trunc: bool, info: dict) -> None:
        """ Update the curriculum with the current step results from the environment.

        :param obs: Observation from teh environment
        :param rew: Reward from the environment
        :param term: True if the episode ended on this step, False otherwise
        :param trunc: True if the episode was truncated on this step, False otherwise
        :param info: Extra information from the environment
        :raises NotImplementedError:
        """
        raise NotImplementedError("This curriculum does not require step updates. Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def update_on_step_batch(self, step_results: List[typing.Tuple[int, int, int, int]]) -> None:
        """Update the curriculum with a batch of step results from the environment.

        This method can be overridden to provide a more efficient implementation. It is used
        as a convenience function and to optimize the multiprocessing message passing throughput.

        :param step_results: List of step results
        """
        for step_result in step_results:
            self.update_on_step(*step_result)

    def update_on_episode(self, episode_return: float, trajectory: List = None) -> None:
        """Update the curriculum with episode results from the environment.

        :param episode_return: Episodic return
        :param trajectory: trajectory of (s, a, r, s, ...), defaults to None
        :raises NotImplementedError:
        """
        raise NotImplementedError("Not yet implemented.")

    def update_on_demand(self, metrics: Dict):
        """Update the curriculum with arbitrary inputs.

        :param metrics: Arbitrary dictionary of information. Can be used to provide gradient/error based
                        updates from the training process.
        :raises NotImplementedError:
        """
        raise NotImplementedError

    # TODO: Move to curriculum sync wrapper?
    def update(self, update_data: typing.Dict[str, tuple]):
        """Update the curriculum with the specified update type.
        TODO: Change method header to not use dictionary, use enums?

        :param update_data: Dictionary
        :type update_data: Dictionary with "update_type" key which maps to one of ["step", "step_batch", "episode", "on_demand", "task_progress", "add_task", "noop"] and "args" with a tuple of the appropriate arguments for the given "update_type".
        :raises NotImplementedError:
        """

        update_type = update_data["update_type"]
        args = update_data["metrics"]

        if update_type == "step":
            self.update_on_step(*args)
        elif update_type == "step_batch":
            self.update_on_step_batch(*args)
        elif update_type == "episode":
            self.update_on_episode(*args)
        elif update_type == "on_demand":
            # Directly pass metrics without expanding
            self.update_on_demand(args)
        elif update_type == "task_progress":
            self.update_task_progress(*args)
        elif update_type == "add_task":
            self.add_task(args)
        elif update_type == "noop":
            # Used to request tasks from the synchronization layer
            pass
        else:
            raise NotImplementedError(f"Update type {update_type} not implemented.")
        self.n_updates += 1

    def update_batch(self, update_data: List[Dict]):
        """Update the curriculum with batch of updates.

        :param update_data: List of updates or potentially varying types
        """
        for update in update_data:
            self.update(update)

    def _sample_distribution(self) -> List[float]:
        """Returns a sample distribution over the task space.

        Any curriculum that maintains a true probability distribution should implement this method to retrieve it.
        """
        raise NotImplementedError

    def _should_use_startup_sampling(self) -> bool:
        return self.warmup_strategy != "none" and self.sampled_tasks < self.warmup_tasks
    
    def _startup_sample(self, k: int) -> List:
        sampled_tasks = []
        if self.warmup_strategy == "fix":
            end_index = min(self.sampled_tasks + k, self.num_tasks)
            sampled_tasks = self.tasks[self.sampled_tasks:end_index]
            self.sampled_tasks = end_index

        if self.warmup_strategy == "random":
            num_tasks = self.num_tasks
            if k > num_tasks:
                raise ValueError("k cannot be larger than the total number of tasks")
            indices = random.sample(range(num_tasks), k)  
            sampled_tasks = [self.tasks[idx] for idx in indices]

        return sampled_tasks
        
    def sample(self, k: int = 1) -> Union[List, Any]:
        """Sample k tasks from the curriculum.

        :param k: Number of tasks to sample, defaults to 1
        :return: Either returns a single task if k=1, or a list of k tasks
        """
        assert self.num_tasks > 0, "Task space is empty. Please add tasks to the curriculum before sampling."

        if self._should_use_startup_sampling():
            tasks = self._startup_sample(k)
            self.sampled_tasks += len(tasks)
            if len(tasks) < k:
                tasks += self.sample(k=k-len(tasks))
            return tasks
        else:
            task_dist = self._sample_distribution()

        # Use list of indices because np.choice does not play nice with tuple tasks
        tasks = self.tasks
        n_tasks = len(tasks)
        task_idx = np.random.choice(list(range(n_tasks)), size=k, p=task_dist)
        return [tasks[i] for i in task_idx]

    def log_metrics(self, writer, step=None, log_full_dist=False):
        """Log the task distribution to the provided tensorboard writer.

        :param writer: Tensorboard summary writer.
        """
        try:
            import wandb
            task_dist = self._sample_distribution()
            if len(task_dist) > 10 and not log_full_dist:
                warnings.warn("Only logging stats for 10 tasks.")
                task_dist = task_dist[:10]
            if self.task_names:
                for idx, prob in enumerate(task_dist):
                    writer.add_scalar(f"curriculum/task_{self.task_space.task_name(idx)}_prob", prob, step)
            else:
                for idx, prob in enumerate(task_dist):
                    writer.add_scalar(f"curriculum/task_{idx}_prob", prob, step)
        except ImportError:
            warnings.warn("Wandb is not installed. Skipping logging.")
        except wandb.errors.Error:
            # No need to crash over logging :)
            warnings.warn("Failed to log curriculum stats to wandb.")
