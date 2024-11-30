import typing
from typing import Any, List, Union

import numpy as np
from gymnasium.spaces import Box

from syllabus.core import Curriculum


class SimpleBoxCurriculum(Curriculum):
    """
    Base class and API for defining curricula to interface with Gym environments.
    """
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self,
                 *curriculum_args,
                 steps: int = 5,
                 success_threshold: float = 0.75,
                 required_successes: int = 10,
                 **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        assert isinstance(self.task_space.gym_space, Box), "SimpleBoxCurriculum only supports Box task spaces."

        self.success_threshold = success_threshold
        self.required_successes = required_successes

        full_range = self.task_space.gym_space.high[1] - self.task_space.gym_space.low[0]
        midpoint = self.task_space.gym_space.low[0] + (full_range / 2.0)
        self.step_size = (full_range / 2.0) / steps
        self.max_range = np.array([midpoint - self.step_size, midpoint + self.step_size])
        self.consecutive_successes = 0
        self.max_reached = False

    def update_task_progress(self, task: typing.Any, progress: Union[float, bool], env_id: int = None) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        if self.max_reached:
            return

        # Check if this task passed success threshold
        if progress > self.success_threshold:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0

        # If we have enough successes in a row, update task
        if self.consecutive_successes >= self.required_successes:
            new_low = max(self.max_range[0] - self.step_size, self.task_space.gym_space.low[0])
            new_high = min(self.max_range[1] + self.step_size, self.task_space.gym_space.high[1])
            self.max_range = np.array([new_low, new_high])
            self.consecutive_successes = 0
            if new_low == self.task_space.gym_space.low[0] and new_high == self.task_space.gym_space.high[1]:
                self.max_reached = True

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        return [self.max_range for _ in range(k)]

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """ Log metrics to writer. """
        logs = [] if logs is None else logs
        logs.append(("range_min", self.max_range[0]))
        logs.append(("range_max", self.max_range[1]))
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)
