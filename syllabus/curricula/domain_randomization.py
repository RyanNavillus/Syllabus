from typing import Any, List

import numpy as np

from syllabus.core import Curriculum


class DomainRandomization(Curriculum):
    """A simple but strong baseline for curriculum learning that uniformly samples a task from the task space."""
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        return [1.0 / self.num_tasks for _ in range(self.num_tasks)]


class BatchedDomainRandomization(Curriculum):
    """A simple but strong baseline for curriculum learning that uniformly samples a task from the task space."""
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, batch_size: int, task_space, warmup_batches: int = 5, **kwargs):
        super().__init__(task_space, **kwargs)
        self.batch_size = batch_size
        self.current_task = None
        self._batch_steps = batch_size  # Start by sampling new task
        self._batch_count = 0
        self.warmup_batches = warmup_batches
        self.distribution = [1.0 / self.num_tasks for _ in range(self.num_tasks)]   # Uniform distribution

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        return self.distribution

    def sample(self, k: int = 1) -> Any:
        tasks = None
        if self._batch_count < self.warmup_batches:
            tasks = super().sample(k=k)

        if self._batch_steps >= self.batch_size:
            self.current_task = super().sample(k=1)
            self._batch_steps -= self.batch_size
            self._batch_count += 1

        if tasks is None:
            tasks = [self.current_task[0] for _ in range(k)]
        return tasks

    def update_on_episode(self, episode_return, length, task, progress, env_id: int = None) -> None:
        super().update_on_episode(episode_return, length, task, progress, env_id=env_id)
        self._batch_steps += length


class SyncedBatchedDomainRandomization(Curriculum):
    """A simple but strong baseline for curriculum learning that uniformly samples a task from the task space."""
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = True

    def __init__(self, batch_size: int, task_space, warmup_batches: int = 1, uniform_chance: float = 0.05, **kwargs):
        super().__init__(task_space, **kwargs)
        self.batch_size = batch_size
        self.warmup_batches = warmup_batches
        self.uniform_chance = uniform_chance

        self.current_task = None
        self._batch_count = 0
        self._should_update = True
        self.distribution = [1.0 / self.num_tasks for _ in range(self.num_tasks)]   # Uniform distribution

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        return self.distribution

    def sample(self, k: int = 1) -> Any:
        """ Sample k tasks from the curriculum."""
        tasks = None
        if self._batch_count < self.warmup_batches:
            tasks = super().sample(k=k)

        if self._should_update:
            self.current_task = super().sample(k=1)[0]
            self._should_update = False

        if tasks is None:
            tasks = []
            for _ in range(k):
                if self.uniform_chance < np.random.rand():
                    tasks.append(self.current_task)
                else:
                    tasks.append(np.random.choice(self.num_tasks))
        return tasks

    def update_batch(self):
        """ Update the current batch."""
        self._should_update = True
        self._batch_count += 1
