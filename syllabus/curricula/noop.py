import typing
from typing import Any, List, Union
from gym.spaces import Box
from syllabus.core import Curriculum


class NoopCurriculum(Curriculum):
    """
    Used to to test API without a curriculum.
    """
    def __init__(self, *curriculum_args, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)

    def _on_step(self, obs, rew, done, info) -> None:
        pass

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        return [self.task_space.sample() for _ in range(k)]
