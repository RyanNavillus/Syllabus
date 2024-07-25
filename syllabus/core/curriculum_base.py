import typing
import warnings
from typing import Any, Callable, List, Tuple, Union

import numpy as np
from gymnasium.spaces import Dict, Box
import random
from syllabus.task_space import TaskSpace
from itertools import product
from .stat_recorder import StatRecorder


# TODO: Move non-generic logic to Uniform class. Allow subclasses to call super for generic error handling
class Curriculum:
    """Base class and API for defining curricula to interface with Gym environments.
    """

    def __init__(self, task_space: TaskSpace, random_start_tasks: int = 0, task_names: Callable = None, record_stats: bool = False, warmup_strategy: str = None, warmup_samples: int = 0) -> None:
        """Initialize the base Curriculum

        :param task_space: the environment's task space from which new tasks are sampled
        :param random_start_tasks: Number of tasks to sample randomly at the start, defaults to 0
        :param task_names: Names of the tasks in the task space, defaults to None
        :param record_stats: Boolean to indicate if statistics should be recorded, defaults to False
        :param warmup_strategy: Strategy for warmup, defaults to None
        :param warmup_samples: Number of warmup samples, defaults to 0
        """
        assert isinstance(task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        self.task_space = task_space
        self.completed_tasks = 0
        self.task_names = task_names if task_names is not None else lambda task, idx: idx
        self.n_updates = 0
        self.startup_sampled_tasks = 0
        self.warmup_strategy = warmup_strategy
        self.warmup_tasks = warmup_samples
        self.fix_curr_index = 0
        self.stat_recorder = StatRecorder(self.task_space, task_names=task_names) if record_stats else None

        if warmup_strategy == "fix" and isinstance(self.task_space.gym_space, Box):
            self.fix_box_space = self._initialize_fixed_grid()

        if self.num_tasks is None:
            warnings.warn("Task space is continuous. Number of warmup tasks can't be compared to the task space size.")
        elif self.num_tasks == 0:
            warnings.warn("Task space is empty. This will cause errors during sampling if no tasks are added.")
        elif warmup_samples > self.num_tasks:
            warnings.warn("Number of warmup tasks is larger than task space, some tasks will be replayed during warmup.")

    @property
    def requires_step_updates(self) -> bool:
        """Returns whether the curriculum requires step updates from the environment.

        :return: True if the curriculum requires step updates, False otherwise
        """
        return self.__class__.REQUIRES_STEP_UPDATES

    @property
    def requires_episode_updates(self) -> bool:
        """Returns whether the curriculum requires episode updates from the environment.

        :return: True if the curriculum requires episode updates, False otherwise
        """
        return self.__class__.REQUIRES_EPISODE_UPDATES

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

    def update_task_progress(self, task: typing.Any, progress: Tuple[float, bool], env_id: int = None) -> None:
        """Update the curriculum with a task and its progress.

        :param task: Task for which progress is being updated.
        :param progress: Progress toward completion or success rate of the given task. 1.0 or True typically indicates a complete task.
        """

        self.completed_tasks += 1

    def update_on_step(self, task: typing.Any, obs: typing.Any, rew: float, term: bool, trunc: bool, info: dict, env_id: int = None) -> None:
        """ Update the curriculum with the current step results from the environment.

        :param obs: Observation from teh environment
        :param rew: Reward from the environment
        :param term: True if the episode ended on this step, False otherwise
        :param trunc: True if the episode was truncated on this step, False otherwise
        :param info: Extra information from the environment
        :raises NotImplementedError:
        """
        raise NotImplementedError("This curriculum does not require step updates. Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def update_on_step_batch(self, step_results: List[typing.Tuple[Any, Any, int, int, int, int]], env_id: int = None) -> None:
        """Update the curriculum with a batch of step results from the environment.

        This method can be overridden to provide a more efficient implementation. It is used
        as a convenience function and to optimize the multiprocessing message passing throughput.

        :param step_results: List of step results
        """
        tasks, obs, rews, terms, truncs, infos = tuple(step_results)
        for i in range(len(obs)):
            self.update_on_step(tasks[i], obs[i], rews[i], terms[i], truncs[i], infos[i], env_id=env_id)

    def update_on_episode(self, episode_return: float, episode_length: int, episode_task: Any, env_id: int = None) -> None:
        """Update the curriculum with episode results from the environment.

        :param episode_return: Episodic return
        :param trajectory: trajectory of (s, a, r, s, ...), defaults to None
        :raises NotImplementedError:
        """
        if self.stat_recorder is not None:
            self.stat_recorder.record(episode_return, episode_length, episode_task, env_id)

    def normalize(self, reward, task):
        """
        Normalize reward by task.
        """
        assert self.stat_recorder is not None, "Curriculum must be initialized with record_stats=True to use normalize()"
        return self.stat_recorder.normalize(reward, task)

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
        env_id = update_data["env_id"] if "env_id" in update_data else None

        if update_type == "step":
            self.update_on_step(*args, env_id=env_id)
        elif update_type == "step_batch":
            self.update_on_step_batch(*args, env_id=env_id)
        elif update_type == "episode":
            self.update_on_episode(*args, env_id=env_id)
        elif update_type == "on_demand":
            # Directly pass metrics without expanding
            self.update_on_demand(args)
        elif update_type == "task_progress":
            self.update_task_progress(*args, env_id=env_id)
        elif update_type == "task_progress_batch":
            tasks, progresses = args
            for task, progress in zip(tasks, progresses):
                self.update_task_progress(task, progress, env_id=env_id)
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
    
    def _initialize_fixed_grid(self):
        dims = self.task_space.gym_space.shape[0]
        samples_per_dim = int(round(pow(self.warmup_tasks,(1 / dims))))
        ranges = [np.linspace(self.task_space.gym_space.low[i], self.task_space.gym_space.high[i], samples_per_dim)
                  for i in range(dims)]
        all_points = list(product(*ranges))
        sampled_tasks = [tuple(point) for point in all_points]

        return sampled_tasks

    def _should_use_startup_sampling(self) -> bool:  
        return self.warmup_strategy != "none" and self.startup_sampled_tasks < self.warmup_tasks
    
    def _startup_sample(self, k: int) -> List:
        sampled_tasks = []

        if isinstance(self.task_space.gym_space, Box):
            if self.warmup_strategy == "fix":
                sampled_tasks = self.fix_box_space
                self.fix_curr_index = (self.fix_curr_index + self.warmup_tasks) % len(sampled_tasks)
            elif self.warmup_strategy == "random":
                sampled_tasks = [self.task_space.gym_space.sample() for _ in range(k)]
    
        else:
            if self.warmup_strategy == "fix":
                if self.fix_curr_index + k > self.num_tasks:
                    sampled_tasks = self.tasks[self.fix_curr_index:self.num_tasks]
                    self.fix_curr_index = self.fix_curr_index + k - self.num_tasks
                    sampled_tasks.extend(self.tasks[0:(self.fix_curr_index)])
                else:
                    sampled_tasks = self.tasks[self.fix_curr_index:self.fix_curr_index + k]
                    self.fix_curr_index += k

            elif self.warmup_strategy == "random":
                # Allows sampling with replacement, making duplicates possible if k > num_tasks.
                indices = random.choices(range(self.num_tasks), k=k)
                sampled_tasks = [self.tasks[idx] for idx in indices]
                
        self.startup_sampled_tasks += k
        return sampled_tasks

    def sample(self, k: int = 1) -> Union[List, Any]:
        """Sample k tasks from the curriculum.

        :param k: Number of tasks to sample, defaults to 1
        :return: Either returns a single task if k=1, or a list of k tasks
        """
        # assert self.num_tasks > 0, "Task space is empty. Please add tasks to the curriculum before sampling."

        if self._should_use_startup_sampling():
            tasks = self._startup_sample(k)
            # Check if the startup sampling has satisfied the request or if there's no progress (no tasks returned)
            if len(tasks) > 0 and len(tasks) < k:  # Check if we need to add more tasks
                additional_tasks = self.sample(k=k-len(tasks))
                tasks.extend(additional_tasks) 
            return tasks
        
        task_dist = self._sample_distribution()

        # Normal sampling process
        tasks = self.tasks
        n_tasks = len(tasks)
        task_idx = np.random.choice(range(n_tasks), size=k, p=task_dist)
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
            log_data = []
            for idx, prob in enumerate(task_dist):
                name = self.task_names(self.tasks[idx], idx)
                log_data.append((f"curriculum/{name}_prob", prob, step))
            for name, prob, step in log_data:
                if writer == wandb:
                    writer.log({name: prob}, step=step)
                else:
                    writer.add_scalar(name, prob, step)
        except ImportError:
            warnings.warn("Wandb is not installed. Skipping logging.")
        except wandb.errors.Error:
            # No need to crash over logging :)
            warnings.warn("Failed to log curriculum stats to wandb.")
        if self.stat_recorder is not None:
            self.stat_recorder.log_metrics(writer, step=step)
