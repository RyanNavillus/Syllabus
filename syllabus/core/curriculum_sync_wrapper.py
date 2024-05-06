import threading
import time
from functools import wraps
from multiprocessing.shared_memory import ShareableList
from typing import List, Tuple

import ray
from torch.multiprocessing import Lock, SimpleQueue
from torch.utils.tensorboard import SummaryWriter

from syllabus.core import Curriculum, decorate_all_functions


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

    @property
    def requires_episode_updates(self):
        return self.curriculum.requires_episode_updates

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

    def update_on_episode(self, episode_return, episode_length, episode_task, env_id=None):
        self.curriculum.update_on_episode(episode_return, episode_length, episode_task, env_id=env_id)

    def update(self, metrics):
        self.curriculum.update(metrics)

    def update_batch(self, metrics):
        self.curriculum.update_batch(metrics)

    def add_task(self, task):
        self.curriculum.add_task(task)

    def normalize(self, rewards, task):
        return self.curriculum.normalize(rewards, task)


class MultiProcessingComponents:
    def __init__(self, task_queue, update_queue):
        self.task_queue = task_queue
        self.update_queue = update_queue
        self._instance_lock = Lock()
        self._env_count = ShareableList([0])
        self._task_count = ShareableList([0])
        self._update_count = ShareableList([0])
        self._debug = False
        self._verbose = False

    def get_id(self):
        with self._instance_lock:
            instance_id = self._env_count[0]
            self._env_count[0] += 1
        return instance_id

    def put_task(self, task):
        self.task_queue.put(task)
        if self._debug:
            task_count = self.added_task()
            if self._verbose:
                print(f"Task added to queue. Task count: {task_count}")

    def get_task(self):
        task = self.task_queue.get()
        if self._debug:
            task_count = self.removed_task()
            if self._verbose:
                print(f"Task removed from queue. Task count: {task_count}")
        return task

    def put_update(self, update):
        self.update_queue.put(update)
        if self._debug:
            update_count = self.added_update()
            if self._verbose:
                print(f"Update added to queue. Update count: {update_count}")

    def get_update(self):
        update = self.update_queue.get()
        if self._debug:
            update_count = self.removed_update()
            if self._verbose:
                print(f"Update removed from queue. Update count: {update_count}")

        return update

    def added_task(self):
        with self._instance_lock:
            self._task_count[0] += 1
            task_count = self._task_count[0]
        return task_count

    def removed_task(self):
        with self._instance_lock:
            self._task_count[0] -= 1
            task_count = self._task_count[0]
        return task_count

    def get_task_count(self):
        with self._instance_lock:
            task_count = self._task_count[0]
        return task_count

    def get_update_count(self):
        with self._instance_lock:
            update_count = self._update_count[0]
        return update_count

    def added_update(self):
        with self._instance_lock:
            self._update_count[0] += 1
            update_count = self._update_count[0]
        return update_count

    def removed_update(self):
        with self._instance_lock:
            self._update_count[0] -= 1
            update_count = self._update_count[0]
        return update_count


class MultiProcessingCurriculumWrapper(CurriculumWrapper):
    def __init__(
        self,
        curriculum: Curriculum,
        task_queue: SimpleQueue,
        update_queue: SimpleQueue,
        sequential_start: bool = True
    ):
        super().__init__(curriculum)
        self.task_queue = task_queue
        self.update_queue = update_queue
        self.sequential_start = sequential_start

        self.update_thread = None
        self.should_update = False
        self.added_tasks = []
        self.num_assigned_tasks = 0

        self._components = MultiProcessingComponents(task_queue, update_queue)

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
        # Process final few updates
        start = time.time()
        end = time.time()
        while end - start < 3 and not self.update_queue.empty():
            time.sleep(0.5)
            end = time.time()

        self.should_update = False
        self.update_thread.join()
        components = self.get_components()
        components._env_count.shm.close()
        components._env_count.shm.unlink()
        components._update_count.shm.close()
        components._update_count.shm.unlink()
        components._task_count.shm.close()
        components._task_count.shm.unlink()
        # components.task_queue.close()
        # components.update_queue.close()

    def _update_queues(self):
        """
        Continuously process completed tasks and sample new tasks.
        """
        # TODO: Refactor long method? Write tests first
        # Update curriculum with environment results:
        while self.should_update:
            requested_tasks = 0
            while not self.update_queue.empty():
                batch_updates = self.get_components().get_update()  # Blocks until update is available

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

                    self.get_components().put_task(message)
                self.num_assigned_tasks += requested_tasks
                time.sleep(0)
            else:
                time.sleep(0.01)

    def update_on_episode(self, episode_return, episode_length, episode_task, env_id=None):
        super().update_on_episode(episode_return, episode_length, episode_task, env_id=env_id)

    def log_metrics(self, writer, step=None):
        super().log_metrics(writer, step=step)
        if isinstance(writer, SummaryWriter):
            if self.get_components()._debug:
                writer.add_scalar("curriculum/updates_in_queue", self.get_components()._update_count[0], step)
                writer.add_scalar("curriculum/tasks_in_queue", self.get_components()._task_count[0], step)

    def add_task(self, task):
        super().add_task(task)
        self.added_tasks.append(task)

    def get_components(self):
        return self._components

    def normalize(self, rewards, task):
        return super().normalize(rewards, task)


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
    return mp_curriculum


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
