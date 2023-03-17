import ray
import time
import threading
from functools import wraps
from typing import List, Tuple

from torch.multiprocessing import SimpleQueue
from syllabus.core import Curriculum, decorate_all_functions


class CurriculumWrapper:
    """
    Wrapper class for adding multiprocessing synchronization to a curriculum.
    """
    def __init__(self, curriculum: Curriculum) -> None:
        self.curriculum = curriculum
        self.task_space = curriculum.task_space
        self.unwrapped = curriculum

    def sample(self, k: int = 1):
        return self.curriculum.sample(k=k)

    def complete_task(self, task, success_prob):
        self.curriculum.complete_task(task, success_prob)

    def _n_tasks(self):
        return self.curriculum._n_tasks()

    def _tasks(self):
        return self.curriculum._tasks()

    def _on_step(self, task, step, reward, done):
        self.curriculum._on_step(task, step, reward, done)

    def log_metrics(self, step=None):
        self.curriculum.log_metrics(step=step)
    def _on_step_batch(self, step_results: List[Tuple[int, int, int, int]]) -> None:
        self.curriculum._on_step_batch(step_results)

    def update_curriculum(self, metrics):
        self.curriculum.update_curriculum(metrics)

    def batch_update_curriculum(self, metrics):
        self.curriculum.batch_update_curriculum(metrics)


class MultiProcessingCurriculumWrapper(CurriculumWrapper):
    """
    Subclass of LearningProgress Curriculum that uses multiprocessing SimpleQueues
    to share tasks and receive feedback from the environment.
    Meant to be used with the MultiprocessingSyncWrapper for Gym environments.
    """
    def __init__(self,
                 curriculum,
                 task_queue: SimpleQueue,
                 update_queue: SimpleQueue):
        super().__init__(curriculum)
        self.task_queue = task_queue
        self.update_queue = update_queue
        self.update_thread = None
        self.should_update = False

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
        while self.should_update:
            # Update curriculum with environment results:
            requested_tasks = 0
            while self.update_queue is not None and not self.update_queue.empty():
                batch_updates = self.update_queue.get()
                if isinstance(batch_updates, dict):
                    batch_updates = [batch_updates]
                # Count updates with "request_sample" set to True
                requested_tasks = sum([result["request_sample"] for result in batch_updates if "request_sample" in result])
                self.batch_update_curriculum(batch_updates)

            # Sample new tasks
            new_tasks = self.curriculum.sample(k=requested_tasks)
            for task in new_tasks:
                self.task_queue.put(task)
            time.sleep(0.00001)

    def __del__(self):
        self.stop()


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
    def __init__(self, curriculum_class, *curriculum_args, actor_name="curriculum", **curriculum_kwargs) -> None:
        sample_curriculum = curriculum_class(*curriculum_args, **curriculum_kwargs)

        super().__init__(sample_curriculum)
        ray_curriculum_class = ray.remote(curriculum_class).options(name=actor_name)
        curriculum = ray_curriculum_class.remote(*curriculum_args, **curriculum_kwargs)
        self.curriculum = curriculum
        self.unwrapped = None
        self.task_space = sample_curriculum.task_space
        del sample_curriculum

    # If you choose to override a function, you will need to forward the call to the remote curriculum.
    # This method is shown here as an example. If you remove it, the same functionality will be provided automatically.
    def sample(self, k: int = 1):
        return ray.get(self.curriculum.sample.remote(k=k))

    def _on_step_batch(self, step_results: List[Tuple[int, int, int, int]]) -> None:
        ray.get(self.curriculum._on_step_batch.remote(step_results))


def make_multiprocessing_curriculum(curriculum_class, *curriculum_args, **curriculum_kwargs):
    """
    Helper function for creating a MultiProcessingCurriculumWrapper.
    """
    curriculum = curriculum_class(*curriculum_args, **curriculum_kwargs)
    task_queue = SimpleQueue()
    update_queue = SimpleQueue()
    mp_curriculum = MultiProcessingCurriculumWrapper(curriculum, task_queue, update_queue)
    mp_curriculum.start()
    return mp_curriculum, task_queue, update_queue


def make_ray_curriculum(curriculum_class, *curriculum_args, actor_name="curriculum", **curriculum_kwargs):
    """
    Helper function for creating a RayCurriculumWrapper.
    """
    return RayCurriculumWrapper(curriculum_class, *curriculum_args, actor_name=actor_name, **curriculum_kwargs)
