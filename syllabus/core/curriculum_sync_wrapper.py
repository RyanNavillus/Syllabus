import gym
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
    def __init__(self, curriculum: Curriculum, batch_results=False) -> None:
        self.curriculum = curriculum
        self.task_space = curriculum.task_space
        self.unwrapped = curriculum

        self.batch_results = batch_results
        if self.batch_results:
            self.batches = []

    def sample(self, k: int = 1):
        return self.curriculum.sample(k=k)

    def complete_task(self, task, success_prob):
        self.curriculum.complete_task(task, success_prob)

    def _n_tasks(self):
        return self.curriculum._n_tasks()

    def _tasks(self):
        return self.curriculum._tasks()

    def on_step(self, task, step, reward, done):
        if self.batch_results:
            self.batches.append((task, step, reward, done))
            if len(self.batches) == 100:
                self.curriculum.on_step_batch(self.batches)
                self.batches = []
        else:
            self.curriculum.on_step(task, step, reward, done)

    def on_step_batch(self, step_results: List[Tuple[int, int, int, int]]) -> None:
        self.curriculum.on_step_batch(step_results)


class MultiProcessingCurriculumWrapper(CurriculumWrapper):
    """
    Subclass of LearningProgress Curriculum that uses multiprocessing SimpleQueues
    to share tasks and receive feedback from the environment.
    Meant to be used with the MultiprocessingSyncWrapper for Gym environments.
    """
    def __init__(self,
                 curriculum,
                 task_queue: SimpleQueue,
                 complete_queue: SimpleQueue,
                 step_queue: SimpleQueue = None,
                 task_space: gym.Space = None,
                 batch_results: bool = False):
        super().__init__(curriculum, batch_results=True)
        self.task_queue = task_queue
        self.complete_queue = complete_queue
        self.step_queue = step_queue
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
            # Process completed tasks
            n_completed_tasks = 0
            while not self.complete_queue.empty():
                task, success_prob = self.complete_queue.get()
                self.curriculum.complete_task(task, success_prob)
                n_completed_tasks += 1

            # Process environment step results:
            while self.step_queue is not None and not self.step_queue.empty():
                batch_results = self.step_queue.get()
                self.on_step_batch(batch_results)

            # Sample new tasks
            new_tasks = self.curriculum.sample(k=n_completed_tasks)
            for task in new_tasks:
                self.task_queue.put(task)
            time.sleep(0.1)


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


@ray.remote(name="curriculum")
def BufferWrapper(CurriculumWrapper):
    pass


@decorate_all_functions(remote_call)
class RayCurriculumWrapper(CurriculumWrapper):
    """
    Subclass of LearningProgress Curriculum that uses Ray to share tasks and receive feedback
    from the environment. The only change is the @ray.remote decorator on the class.

    The @decorate_all_functions(remote_call) annotation automatically forwards all functions not explicitly
    overridden here to the remote curriculum.
    """
    def __init__(self, curriculum_class, *curriculum_args, batch_results=True, **curriculum_kwargs) -> None:
        sample_curriculum = curriculum_class(*curriculum_args, **curriculum_kwargs)
        super().__init__(sample_curriculum, batch_results=batch_results)
        ray_curriculum_class = ray.remote(curriculum_class).options(name="curriculum")
        curriculum = ray_curriculum_class.remote(*curriculum_args, **curriculum_kwargs)
        self.curriculum = curriculum
        self.unwrapped = None

        # Set basic properties here so that wrapper has the same interface as the curriculum.
        self.task_space = sample_curriculum.task_space
        del sample_curriculum

    # If you choose to override a function, you will need to forward the call to the remote curriculum.
    # This method is shown here as an example. If you remove it, the same functionality will be provided automatically.
    def sample(self, k: int = 1):
        return ray.get(self.curriculum.sample.remote(k=k))

    # We override this to prevent an immediate ray.get and instead allow the updates to be batched
    def on_step(self, task, step, reward, done):
        super().on_step(task, step, reward, done)
