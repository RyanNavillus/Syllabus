import copy
import signal
import sys
import threading
import time
import warnings
from functools import wraps
from multiprocessing.shared_memory import ShareableList
from queue import Empty
from typing import Dict

import ray
from torch.multiprocessing import Lock, Queue

from syllabus.core import Curriculum
from syllabus.utils import UsageError, decorate_all_functions


class CurriculumWrapper:
    """Wrapper class for adding multiprocessing synchronization to a curriculum.
    """

    def __init__(self, curriculum: Curriculum) -> None:
        self.curriculum = curriculum
        if hasattr(curriculum, "unwrapped") and curriculum.unwrapped is not None:
            self.unwrapped = curriculum.unwrapped
        else:
            self.unwrapped = curriculum
        self.task_space = self.unwrapped.task_space

    @property
    def num_tasks(self):
        return self.task_space.num_tasks

    def count_tasks(self, task_space=None):
        return self.task_space.count_tasks(gym_space=task_space)

    @property
    def tasks(self):
        return self.task_space.tasks

    @property
    def requires_step_updates(self):
        return self.curriculum.requires_step_updates

    def get_tasks(self, task_space=None):
        return self.task_space.get_tasks(gym_space=task_space)

    def sample(self, k=1):
        return self.curriculum.sample(k=k)

    def update_task_progress(self, task, progress):
        self.curriculum.update_task_progress(task, progress)

    def update_on_step(self, task, obs, reward, term, trunc, info, progress):
        self.curriculum.update_on_step(task, obs, reward, term, trunc, info, progress)

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        return self.curriculum.log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)

    def update_on_step_batch(self, step_results, env_id=None):
        self.curriculum.update_on_step_batch(step_results, env_id=env_id)

    def update_on_episode(self, episode_return, length, task, progress, env_id=None):
        self.curriculum.update_on_episode(episode_return, length, task, progress, env_id=env_id)

    def normalize(self, rewards, task):
        return self.curriculum.normalize(rewards, task)

    def __getattr__(self, attr):
        curriculum_atr = getattr(self.curriculum, attr, None)
        if curriculum_atr is not None:
            return curriculum_atr


class MultiProcessingComponents:
    def __init__(self, requires_step_updates, max_queue_size=1000000, timeout=60, max_envs=None):
        self.requires_step_updates = requires_step_updates
        self.task_queue = Queue(maxsize=max_queue_size)
        self.update_queue = Queue(maxsize=max_queue_size)
        self._instance_lock = Lock()
        self._env_count = ShareableList([0])
        self._debug = True
        self.timeout = timeout
        self.max_envs = max_envs
        self._maxsize = max_queue_size
        self.started = False

    def peek_id(self):
        return self._env_count[0]

    def get_id(self):
        with self._instance_lock:
            instance_id = self._env_count[0]
            self._env_count[0] += 1
        return instance_id

    def should_sync(self, env_id):
        # Only receive step updates from self.max_envs environments
        if self.max_envs is not None and env_id >= self.max_envs:
            return False
        return True

    def put_task(self, task):
        self.task_queue.put(task, block=False)

    def get_task(self):
        try:
            if self.started and self.task_queue.empty():
                warnings.warn(
                    f"Task queue capacity is {self.task_queue.qsize()} / {self.task_queue._maxsize}. Program may deadlock if task_queue is empty. If the update queue capacity is increasing, consider optimizing your curriculum or reducing the number of environments. Otherwise, consider increasing the buffer_size for your environment sync wrapper.")
            task = self.task_queue.get(block=True, timeout=self.timeout)
            return task
        except Empty as e:
            raise UsageError(
                f"Failed to get task from queue after {self.timeout}s. Queue capacity is {self.task_queue.qsize()} / {self.task_queue._maxsize} items.") from e

    def put_update(self, update):
        self.update_queue.put(copy.deepcopy(update), block=False)

    def get_update(self):
        update = self.update_queue.get(block=False)
        return update

    def close(self):
        if self._env_count is not None:
            self._env_count.shm.close()
            try:
                self._env_count.shm.unlink()
            except FileNotFoundError:
                pass    # Already unlinked
            self.task_queue.close()
            self.update_queue.close()
            self._env_count = None

    def get_metrics(self, log_n_tasks=1):
        logs = []
        logs.append(("curriculum/updates_in_queue", self.update_queue.qsize()))
        logs.append(("curriculum/tasks_in_queue", self.task_queue.qsize()))
        return logs


class CurriculumSyncWrapper(CurriculumWrapper):
    def __init__(
        self,
        curriculum: Curriculum,
        **kwargs,
    ):
        super().__init__(curriculum)

        self.update_thread = None
        self.should_update = False
        self.added_tasks = []
        self.num_assigned_tasks = 0

        self.components = MultiProcessingComponents(self.curriculum.requires_step_updates, **kwargs)

    def start(self):
        """
        Start the thread that reads the complete_queue and reads the task_queue.
        """
        if not self.should_update:
            self.update_thread = threading.Thread(name='update', target=self._update_queues, daemon=True)
            self.should_update = True
            self.components.started = True
            signal.signal(signal.SIGINT, self._sigint_handler)
            self.update_thread.start()

    def stop(self):
        """
        Stop the thread that reads the complete_queue and reads the task_queue.
        """
        self.should_update = False
        self.components.started = False
        self.update_thread.join()
        self.components.close()

    def _sigint_handler(self, sig, frame):
        self.stop()
        sys.exit(0)

    def _update_queues(self):
        """
        Continuously process completed tasks and sample new tasks.
        """
        # Update curriculum with environment results:
        while self.should_update:
            if not self.components.update_queue.empty():
                update = self.components.get_update()  # Blocks until update is available

                if isinstance(update, list):
                    update = update[0]

                # Sample new tasks if requested
                if "request_sample" in update and update["request_sample"]:
                    new_tasks = self.curriculum.sample(k=1)
                    for task in new_tasks:
                        message = {"next_task": task}
                        self.components.put_task(message)
                        self.num_assigned_tasks += 1
                self.route_update(update)
                time.sleep(0.0)
            else:
                time.sleep(0.01)

    def route_update(self, update_data: Dict[str, tuple]):
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
        elif update_type == "task_progress":
            self.update_task_progress(*args, env_id=env_id)
        elif update_type == "noop":
            # Used to request tasks from the synchronization layer
            pass
        else:
            raise NotImplementedError(f"Update type {update_type} not implemented.")

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        logs = [] if logs is None else logs
        logs += self.components.get_metrics(log_n_tasks=log_n_tasks)
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)


def remote_call(func):
    """
    Decorator for automatically forwarding calls to the curriculum via ray remote calls.

    Note that this causes functions to block, and should be only used for operations that do not require parallelization.
    """
    @wraps(func)
    def wrapper(self, *args, **kw):
        f_name = func.__name__
        parent_func = getattr(CurriculumWrapper, f_name)
        child_func = getattr(self, f_name)

        # Only forward call if subclass does not explicitly override the function.
        if child_func == parent_func:
            curriculum_func = getattr(self.curriculum, f_name)
            return ray.get(curriculum_func.remote(*args, **kw))
    return wrapper


def make_multiprocessing_curriculum(curriculum, start=True, **kwargs):
    """
    Helper function for creating a MultiProcessingCurriculumWrapper.
    """
    mp_curriculum = CurriculumSyncWrapper(curriculum, **kwargs)
    if start:
        mp_curriculum.start()
    return mp_curriculum


@ray.remote
class RayCurriculumWrapper(CurriculumWrapper):
    def __init__(self, curriculum: Curriculum) -> None:
        super().__init__(curriculum)

    def get_remote_attr(self, name: str):
        next_obj = getattr(self.curriculum, name)
        return next_obj


@decorate_all_functions(remote_call)
class RayCurriculumSyncWrapper(CurriculumWrapper):
    """
    Subclass of LearningProgress Curriculum that uses Ray to share tasks and receive feedback
    from the environment. The only change is the @ray.remote decorator on the class.

    The @decorate_all_functions(remote_call) annotation automatically forwards all functions not explicitly
    overridden here to the remote curriculum. This is intended to forward private functions of Curriculum subclasses
    for convenience.
    # TODO: Implement the Curriculum methods explicitly
    """

    def __init__(self, curriculum, actor_name="curriculum") -> None:
        super().__init__(curriculum)
        self.curriculum = RayCurriculumWrapper.options(name=actor_name).remote(curriculum)
        self.unwrapped = None
        self.task_space = curriculum.task_space
        self.added_tasks = []

    # If you choose to override a function, you will need to forward the call to the remote curriculum.
    # This method is shown here as an example. If you remove it, the same functionality will be provided automatically.
    def sample(self, k: int = 1):
        return ray.get(self.curriculum.sample.remote(k=k))

    def update_on_step_batch(self, step_results, env_id=None) -> None:
        ray.get(self.curriculum._on_step_batch.remote(step_results))


def make_ray_curriculum(curriculum, actor_name="curriculum", **kwargs):
    """
    Helper function for creating a RayCurriculumWrapper.
    """
    return RayCurriculumSyncWrapper(curriculum, actor_name=actor_name, **kwargs)
