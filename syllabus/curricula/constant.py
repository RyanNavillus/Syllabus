from typing import Any, List, Union

from syllabus.core import Curriculum


class Constant(Curriculum):
    """
    Used to to test API without a curriculum.
    """

    def __init__(self, default_task, *curriculum_args, require_step_updates=False, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self.default_task = self.task_space.encode(default_task)
        self.require_step_updates = require_step_updates

    @property
    def requires_step_updates(self) -> bool:
        return self.require_step_updates

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        return [self.default_task for _ in range(k)]

    def update_task_progress(self, task, progress, env_id=None) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        pass

    def update_on_step(self, task, obs, rew, term, trunc, info, progress, env_id=None) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        pass

    def update_on_step_batch(self, step_results, env_id=None) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        pass

    def update_on_episode(self, episode_return, length, task, progress, env_id=None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        pass

    def _sample_distribution(self, k: int = 1) -> Union[List, Any]:
        """
        Returns a sample distribution over the task space.
        """
        dist = [1.0 / self.num_tasks for _ in range(self.num_tasks)]
        dist[self.default_task] = 1.0
        return dist
