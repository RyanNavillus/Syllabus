import warnings
from typing import Any, Callable, List, Tuple, Union, Dict

import numpy as np

from syllabus.task_space import TaskSpace
from .stat_recorder import StatRecorder


# TODO: Move non-generic logic to Uniform class. Allow subclasses to call super for generic error handling
class Curriculum:
    """Base class and API for defining curricula to interface with Gym environments.
    """
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, task_space: TaskSpace, random_start_tasks: int = 0, task_names: Callable = None, record_stats: bool = False) -> None:
        """Initialize the base Curriculum

        :param task_space: the environment's task space from which new tasks are sampled
        TODO: Implement this in a way that works with any curriculum, maybe as a wrapper
        :param random_start_tasks: Number of uniform random tasks to sample before using the algorithm's sample method, defaults to 0
        TODO: Use task space for this
        :param task_names: Names of the tasks in the task space, defaults to None
        """
        assert isinstance(
            task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        self.task_space = task_space
        self.random_start_tasks = random_start_tasks
        self.completed_tasks = 0
        self.task_names = task_names if task_names is not None else lambda task, idx: idx
        self.n_updates = 0
        self.stat_recorder = StatRecorder(self.task_space, task_names=task_names) if record_stats else None

        if self.num_tasks == 0:
            warnings.warn("Task space is empty. This will cause errors during sampling if no tasks are added.")

    @property
    def requires_step_updates(self) -> bool:
        """Returns whether the curriculum requires step updates from the environment.

        :return: True if the curriculum requires step updates, False otherwise
        """
        return self.__class__.REQUIRES_STEP_UPDATES

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
        return self.task_space._task_list

    def update_task_progress(self, task: Any, progress: Union[float, bool], env_id: int = None) -> None:
        """Update the curriculum with a task and its progress. This is used for binary tasks that can be completed mid-episode.

        :param task: Task for which progress is being updated.
        :param progress: Progress toward completion or success rate of the given task. 1.0 or True typically indicates a complete task.
        :param env_id: Environment identifier
        """
        self.completed_tasks += 1

    def update_on_step(self, task: Any, obs: Any, rew: float, term: bool, trunc: bool, info: dict, progress: Union[float, bool], env_id: int = None) -> None:
        """ Update the curriculum with the current step results from the environment.

        :param obs: Observation from teh environment
        :param rew: Reward from the environment
        :param term: True if the episode ended on this step, False otherwise
        :param trunc: True if the episode was truncated on this step, False otherwise
        :param info: Extra information from the environment
        :param progress: Progress toward completion or success rate of the given task. 1.0 or True typically indicates a complete task.
        :param env_id: Environment identifier
        :raises NotImplementedError:
        """
        raise NotImplementedError(
            "This curriculum does not require step updates. Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def update_on_step_batch(self, step_results: Tuple[List[Any], List[Any], List[int], List[bool], List[bool], List[Dict], List[int]], env_id: int = None) -> None:
        """Update the curriculum with a batch of step results from the environment.

        This method can be overridden to provide a more efficient implementation. It is used
        as a convenience function and to optimize the multiprocessing message passing throughput.

        :param step_results: List of step results
        :param env_id: Environment identifier
        """
        tasks, obs, rews, terms, truncs, infos, progresses = tuple(step_results)
        for t, o, r, te, tr, i, p in zip(tasks, obs, rews, terms, truncs, infos, progresses):
            self.update_on_step(t, o, r, te, tr, i, p, env_id=env_id)

    def update_on_episode(self, episode_return: float, length: int, task: Any, progress: Union[float, bool], env_id: int = None) -> None:
        """Update the curriculum with episode results from the environment.

        :param episode_return: Episodic return
        :param length: Length of the episode
        :param task: Task for which the episode was completed
        :param progress: Progress toward completion or success rate of the given task. 1.0 or True typically indicates a complete task.
        :param env_id: Environment identifier
        :raises NotImplementedError:
        """
        if self.stat_recorder is not None:
            self.stat_recorder.record(episode_return, length, task, env_id)

    def normalize(self, reward, task):
        """
        Normalize reward by task.

        :param reward: Reward to normalize
        :param task: Task for which the reward was received
        :return: Normalized reward
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
    def update(self, update_data: Dict[str, tuple]):
        """Update the curriculum with the specified update type.
        TODO: Change method header to not use dictionary, use enums?

        :param update_data: Dictionary
        :type update_data: Dictionary with "update_type" key which maps to one of ["step", "step_batch", "episode", "on_demand", "task_progress", "noop"] and "args" with a tuple of the appropriate arguments for the given "update_type".
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
        return self.random_start_tasks > 0 and self.completed_tasks < self.random_start_tasks

    def _startup_sample(self) -> List:
        task_dist = [0.0 / self.num_tasks for _ in range(self.num_tasks)]
        task_dist[0] = 1.0
        return task_dist

    def sample(self, k: int = 1) -> Union[List, Any]:
        """Sample k tasks from the curriculum.

        :param k: Number of tasks to sample, defaults to 1
        :return: Either returns a single task if k=1, or a list of k tasks
        """
        # assert self.num_tasks > 0, "Task space is empty. Please add tasks to the curriculum before sampling."

        if self._should_use_startup_sampling():
            return self._startup_sample()

        # Use list of indices because np.choice does not play nice with tuple tasks
        # tasks = self.tasks
        n_tasks = self.num_tasks
        task_dist = self._sample_distribution()
        task_idx = np.random.choice(list(range(n_tasks)), size=k, p=task_dist)
        return task_idx

    def log_metrics(self, writer, step=None, log_full_dist=False):
        """Log the task distribution to the provided tensorboard writer.

        :param writer: Tensorboard summary writer.
        """
        try:
            import wandb
            task_dist = self._sample_distribution()
            if len(task_dist) > 5 and not log_full_dist:
                warnings.warn("Only logging stats for 5 tasks.")
                task_dist = task_dist[:5]
            log_data = []
            for idx, prob in enumerate(task_dist):
                name = self.task_names(self.tasks[idx], idx)
                log_data.append((f"curriculum/{name}_prob", prob, step))
            for name, prob, step in log_data:
                if writer == wandb:
                    writer.log({name: prob, "global_step": step})
                else:
                    writer.add_scalar(name, prob, step)
        except ImportError:
            warnings.warn("Wandb is not installed. Skipping logging.")
        except wandb.errors.Error:
            # No need to crash over logging :)
            warnings.warn("Failed to log curriculum stats to wandb.")
        if self.stat_recorder is not None:
            self.stat_recorder.log_metrics(writer, step=step)
