import typing
from typing import Any, List, Union
from gym.spaces import Box
from syllabus.core import Curriculum


class NoopCurriculum(Curriculum):
    """
    Used to to test API without a curriculum.
    """
    REQUIRES_STEP_UPDATES = True
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, default_task, *curriculum_args, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self.default_task = default_task

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        return [self.default_task for _ in range(k)]
    
    def update_on_complete(self, task, success_prob) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """ 
        pass

    def update_on_step(self, obs, rew, done, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        pass

    def update_on_step_batch(self, step_results) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        pass

    def update_on_episode(self, episode_return) -> None:
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

    def update_curriculum(self, update_data):
        """
        Update the curriculum with the specified update type.
        """
        pass