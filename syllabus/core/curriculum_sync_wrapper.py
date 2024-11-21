import threading
import time
from functools import wraps
from multiprocessing.shared_memory import ShareableList
from typing import List, Tuple

import signal
import ray
from torch.multiprocessing import Lock, Queue
from queue import Empty
from torch.utils.tensorboard import SummaryWriter

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

    def update(self, metrics):
        self.curriculum.update(metrics)

    def update_batch(self, metrics):
        self.curriculum.update_batch(metrics)

    def normalize(self, rewards, task):
        return self.curriculum.normalize(rewards, task)


class MultiProcessingComponents:
    def __init__(self, maxsize=1000000, timeout=60):
        self.task_queue = Queue(maxsize=maxsize)
        self.update_queue = Queue(maxsize=maxsize)
        self._instance_lock = Lock()
        self._env_count = ShareableList([0])
        self._debug = True
        self.timeout = timeout

    def get_id(self):
        with self._instance_lock:
            instance_id = self._env_count[0]
            self._env_count[0] += 1
        return instance_id

    def put_task(self, task):
        try:
            self.task_queue.put(task, timeout=self.timeout)
        except Empty as e:
            raise UsageError(
                f"Failed to put task in queue after {self.timeout}s. Queue capacity is {self.task_queue.qsize()} / {self.task_queue._maxsize} items.") from e

    def get_task(self):
        try:
            task = self.task_queue.get(timeout=self.timeout)
        except Empty as e:
            raise UsageError(
                f"Failed to get task from queue after {self.timeout}s. Queue capacity is {self.task_queue.qsize()} / {self.task_queue._maxsize} items.") from e
        return task

    def put_update(self, update):
        try:
            self.update_queue.put(update, timeout=self.timeout)
        except Empty as e:
            raise UsageError(
                f"Failed to put update in queue after {self.timeout}s. Queue capacity is {self.update_queue.qsize()} / {self.update_queue._maxsize} items.") from e

    def get_update(self):
        try:
            update = self.update_queue.get(timeout=self.timeout)
        except Empty as e:
            raise UsageError(
                f"Failed to get update from queue after {self.timeout}s. Queue capacity is {self.update_queue.qsize()} / {self.update_queue._maxsize} items.") from e

        return update

    def close(self):
        if self._env_count is not None:
            self._env_count.shm.close()
            self._env_count.shm.unlink()
            self.task_queue.close()
            self.update_queue.close()
            self._env_count = None
            del self.task_queue
            del self.update_queue

    def get_metrics(self, log_n_tasks=1):
        logs = []
        logs.append(("curriculum/updates_in_queue", self.update_queue.qsize()))
        logs.append(("curriculum/tasks_in_queue", self.task_queue.qsize()))
        return logs


class MultiProcessingCurriculumWrapper(CurriculumWrapper):
    def __init__(
        self,
        curriculum: Curriculum,
        sequential_start: bool = True,
        max_queue_size: int = 1000000,
        timeout: int = 60,
    ):
        super().__init__(curriculum)
        self.sequential_start = sequential_start

        self.update_thread = None
        self.should_update = False
        self.added_tasks = []
        self.num_assigned_tasks = 0

        self.components = MultiProcessingComponents(maxsize=max_queue_size, timeout=timeout)

    def start(self):
        """
        Start the thread that reads the complete_queue and reads the task_queue.
        """
        self.update_thread = threading.Thread(name='update', target=self._update_queues, daemon=True)
        self.should_update = True
        signal.signal(signal.SIGINT, self._sigint_handler)
        self.update_thread.start()

    def stop(self):
        """
        Stop the thread that reads the complete_queue and reads the task_queue.
        """
        self.should_update = False
        self.update_thread.join()
        self.components.close()

    def _sigint_handler(self, sig, frame):
        self.stop()

    def _update_queues(self):
        """
        Continuously process completed tasks and sample new tasks.
        """
        # TODO: Refactor long method? Write tests first
        # Update curriculum with environment results:
        while self.should_update:
            requested_tasks = 0
            while not self.components.update_queue.empty():
                batch_updates = self.components.get_update()  # Blocks until update is available

                if isinstance(batch_updates, dict):
                    batch_updates = [batch_updates]

                # Count number of requested tasks
                for update in batch_updates:
                    if "request_sample" in update and update["request_sample"]:
                        requested_tasks += 1

                self.update_batch(batch_updates)

            # Sample new tasks
            if requested_tasks > 0:
                new_tasks = self.curriculum.sample(k=requested_tasks)
                for i, task in enumerate(new_tasks):
                    message = {
                        "next_task": task,
                        "sample_id": self.num_assigned_tasks + i,
                    }
                    self.components.put_task(message)
                self.num_assigned_tasks += requested_tasks
                time.sleep(0)
            else:
                time.sleep(0.01)

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        logs = [] if logs is None else logs
        logs += self.components.get_metrics(log_n_tasks=log_n_tasks)
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)

    def __del__(self):
        self.stop()
        del self


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
    mp_curriculum = MultiProcessingCurriculumWrapper(curriculum, **kwargs)
    if start:
        mp_curriculum.start()
    return mp_curriculum


@ray.remote
class RayWrapper(CurriculumWrapper):
    def __init__(self, curriculum: Curriculum) -> None:
        super().__init__(curriculum)

    def get_remote_attr(self, name: str):
        next_obj = getattr(self.curriculum, name)
        return next_obj


@decorate_all_functions(remote_call)
class RayCurriculumWrapper(CurriculumWrapper):
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
        self.curriculum = RayWrapper.options(name=actor_name).remote(curriculum)
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
    return RayCurriculumWrapper(curriculum, actor_name=actor_name, **kwargs)
