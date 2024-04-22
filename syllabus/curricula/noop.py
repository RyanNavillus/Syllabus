from typing import Any, List, Union

from syllabus.core import Curriculum


class NoopCurriculum(Curriculum):
    """
    Used to to test API without a curriculum.
    """
    REQUIRES_STEP_UPDATES = True
    REQUIRES_EPISODE_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, default_task, *curriculum_args, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self.default_task = self.task_space.encode(default_task)

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        return [self.default_task for _ in range(k)]

    def update_task_progress(self, task, success_prob, env_id: int = None) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        pass

    def update_on_step(self, task, obs, rew, term, trunc, info, env_id: int = None) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        pass

    def update_on_step_batch(self, step_results, env_id: int = None) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        pass

    def update_on_episode(self, episode_return, episode_length, episode_task, env_id: int = None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        pass

    def update_on_demand(self, metrics):
        """
        Update the curriculum with arbitrary inputs.
        """
        pass

    def add_task(self, task: tuple) -> None:
        pass

    def update(self, update_data):
        """
        Update the curriculum with the specified update type.
        """
        pass
