import threading
from functools import wraps
from typing import List, Tuple

import ray
import time
from torch.multiprocessing import SimpleQueue

from syllabus.core import Curriculum, decorate_all_functions


class CurriculumWrapper:
    """Wrapper class for adding multiprocessing synchronization to a curriculum.
    """
    def __init__(self, curriculum: Curriculum) -> None:
        self.curriculum = curriculum
        self.task_space = curriculum.task_space
        self.unwrapped = curriculum

    @property
    def num_tasks(self):
        return self.task_space.num_tasks

    def count_tasks(self, task_space=None):
        return self.task_space.count_tasks(gym_space=task_space)

    @property
    def tasks(self):
        return self.task_space.tasks   

    def get_tasks(self, task_space=None):
        return self.task_space.get_tasks(gym_space=task_space)

    def sample(self, k=1):
        return self.curriculum.sample(k=k)

    def update_task_progress(self, task, progress):
        self.curriculum.update_task_progress(task, progress)

    def update_on_step(self, task, step, reward, term, trunc):
        self.curriculum.update_on_step(task, step, reward, term, trunc)

    def log_metrics(self, writer, step=None):
        self.curriculum.log_metrics(writer, step=step)

    def update_on_step_batch(self, step_results):
        self.curriculum.update_on_step_batch(step_results)

    def update(self, metrics):
        self.curriculum.update(metrics)

    def batch_update(self, metrics):
        self.curriculum.update_batch(metrics)

    def add_task(self, task):
        self.curriculum.add_task(task)


class MultiProcessingCurriculumWrapper(CurriculumWrapper):
    """Wrapper which sends tasks and receives updates from environments wrapped in a corresponding MultiprocessingSyncWrapper.
    """
    def __init__(self,
                 curriculum: Curriculum,
                 task_queue: SimpleQueue,
                 update_queue: SimpleQueue,
                 sequential_start: bool = True):
        super().__init__(curriculum)
        self.task_queue = task_queue
        self.update_queue = update_queue
        self.update_thread = None
        self.should_update = False
        self.added_tasks = []
        self.num_assigned_tasks = 0
        # TODO: Check if task_space is enumerable
        self.sequential_start = sequential_start

    def start(self):
        """
        Start the thread that reads the complete_queue and reads the task_queue.
        """
        self.update_thread = threading.Thread(name='update', target=self._update_queues, daemon=True)
        self.should_update = True
        self.update_thread.start()

    def stop(self):
        """
        Stop the thread that reads the complete_queue and reads the task_queue.
        """
        self.should_update = False

    def _update_queues(self):
        """
        Continuously process completed tasks and sample new tasks.
        """
        # TODO: Refactor long method? Write tests first
        while self.should_update:
            # Update curriculum with environment results:
            requested_tasks = 0
            while not self.update_queue.empty():
                batch_updates = self.update_queue.get()
                if isinstance(batch_updates, dict):
                    batch_updates = [batch_updates]
                for update in batch_updates:
                    # Count updates with "request_sample" set to True
                    if "request_sample" in update and update["request_sample"]:
                        requested_tasks += 1
                    # Decode task and task progress
                    if update["update_type"] == "task_progress":
                        update["metrics"] = (self.task_space.decode(update["metrics"][0]), update["metrics"][1])
                self.batch_update(batch_updates)

            # Sample new tasks
            if requested_tasks > 0:
                # TODO: Move this to curriculum, not sync wrapper
                # Sequentially sample task_space before using curriculum method
                if (self.sequential_start and
                        self.task_space.num_tasks is not None and
                        self.num_assigned_tasks + requested_tasks < self.task_space.num_tasks):
                    # Sample unseen tasks sequentially before using curriculum method
                    new_tasks = self.task_space.list_tasks()[self.num_assigned_tasks:self.num_assigned_tasks + requested_tasks]
                else:
                    new_tasks = self.curriculum.sample(k=requested_tasks)
                for i, task in enumerate(new_tasks):
                    message = {
                        "next_task": self.task_space.encode(task),
                        "added_tasks": self.added_tasks,
                        "sample_id": self.num_assigned_tasks + i,
                    }
                    self.task_queue.put(message)
                    self.added_tasks = []
                self.num_assigned_tasks += requested_tasks
            time.sleep(0.01)

    def __del__(self):
        self.stop()

    def log_metrics(self, writer, step=None):
        super().log_metrics(writer, step=step)
        writer.add_scalar("curriculum/requested_tasks", self.num_assigned_tasks, step)

    def add_task(self, task):
        super().add_task(task)
        self.added_tasks.append(task)


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


def make_multiprocessing_curriculum(curriculum, **kwargs):
    """
    Helper function for creating a MultiProcessingCurriculumWrapper.
    """
    task_queue = SimpleQueue()
    update_queue = SimpleQueue()
    mp_curriculum = MultiProcessingCurriculumWrapper(curriculum, task_queue, update_queue, **kwargs)
    mp_curriculum.start()
    return mp_curriculum, task_queue, update_queue


@ray.remote
class RayWrapper(CurriculumWrapper):
    def __init__(self, curriculum: Curriculum) -> None:
        super().__init__(curriculum)


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

    def update_on_step_batch(self, step_results: List[Tuple[int, int, int, int]]) -> None:
        ray.get(self.curriculum._on_step_batch.remote(step_results))

    def add_task(self, task):
        super().add_task(task)
        self.added_tasks.append(task)


def make_ray_curriculum(curriculum, actor_name="curriculum", **kwargs):
    """
    Helper function for creating a RayCurriculumWrapper.
    """
    return RayCurriculumWrapper(curriculum, actor_name=actor_name, **kwargs)
