from typing import Any, List

from syllabus.core import Curriculum


class DomainRandomization(Curriculum):
    """A simple but strong baseline for curriculum learning that uniformly samples a task from the task space.
    """
    REQUIRES_STEP_UPDATES = False
    REQUIRES_EPISODE_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        return [1.0 / self.num_tasks for _ in range(self.num_tasks)]

    def add_task(self, task: Any) -> None:
        self.task_space.add_task(task)


class BatchedDomainRandomization(Curriculum):
    """A simple but strong baseline for curriculum learning that uniformly samples a task from the task space.
    """
    REQUIRES_STEP_UPDATES = False
    REQUIRES_EPISODE_UPDATES = True
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, batch_size: int, task_space, **kwargs):
        super().__init__(task_space, **kwargs)
        self.batch_size = batch_size
        self.current_task = None
        self._batch_steps = batch_size  # Start by sampling new task
        self.distribution = [1.0 / self.num_tasks for _ in range(self.num_tasks)]

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        return self.distribution

    def sample(self, k: int = 1) -> Any:
        if self._batch_steps >= self.batch_size:
            # Uniform distribution
            self.current_task = super().sample(k=1)
            self._batch_steps -= self.batch_size
        return [self.current_task[0] for _ in range(k)]

    def update_on_episode(self, episode_returns, episode_length, episode_task, env_id: int = None) -> None:
        super().update_on_episode(episode_returns, episode_length, episode_task, env_id=env_id)
        self._batch_steps += episode_length
