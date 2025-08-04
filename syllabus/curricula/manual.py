from typing import Any, List, Union

from syllabus.core import Curriculum
from syllabus.utils import UsageError


class Manual(Curriculum):
    """
    Used to to test API without a curriculum.
    """

    def __init__(self, *curriculum_args, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self.task_queue = []

    @property
    def requires_step_updates(self) -> bool:
        return False

    def add_tasks(self, tasks: List[Any]) -> None:
        """
        Add tasks to the manual curriculum.
        """
        if not isinstance(tasks, list):
            raise UsageError("Tasks must be provided as a list.")
        self.task_queue.extend(tasks)

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        try:
            print(self.task_queue)
            return [self.task_queue.pop(0) for _ in range(k)]
        except IndexError:
            raise UsageError("Manual curriculum ran out of tasks. Add more tasks with add_tasks() before sampling.")

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
