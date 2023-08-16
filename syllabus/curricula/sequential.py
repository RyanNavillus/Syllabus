import numpy as np
from gym.spaces import Tuple, Dict
from typing import Any, List, Union
from syllabus.core import Curriculum


class SequentialCurriculum(Curriculum):
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, task_list: List[Any], *curriculum_args, num_repeats: List[int] = None, repeat_list=True, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self.task_list = task_list
        self.num_repeats = num_repeats if num_repeats is not None else [1] * len(task_list)
        self.repeat_list = repeat_list
        self._task_index = 0
        self._repeat_index = 0

    def _sample_distribution(self) -> List[float]:
        """
        Return None to indicate that tasks are not drawn from a distribution.
        """
        return None

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Choose the next k tasks from the list.
        """
        tasks = []
        for _ in range(k):
            # Check if there are any tasks left to sample from
            if self._task_index >= len(self.task_list):
                self._task_index = 0
                if not self.repeat_list:
                    raise ValueError(f"Ran out of tasks to sample from. {sum(self.num_repeats)} sampled")

            # Sample the next task and increment index
            tasks.append(self.task_list[self._task_index])
            self._repeat_index += 1

            # Check if we need to repeat the current task
            if self._repeat_index >= self.num_repeats[self._task_index]:
                self._task_index += 1
                self._repeat_index = 0
        return tasks
    
    def remaining_tasks(self):
        """
        Return the number of tasks remaining in the manual curriculum.
        """
        if self._task_index >= len(self.task_list):
            return 0
        return (self.num_repeats[self._task_index] - self._repeat_index) + sum(repeat for repeat in self.num_repeats[self._task_index+1:])

