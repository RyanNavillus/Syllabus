from typing import Any, List, SupportsFloat, SupportsInt, Union

import numpy as np
from gymnasium.spaces import Box

from syllabus.core import Curriculum


class SimulatedAnnealing(Curriculum):
    """ Curriculum that anneals a set of values from start_values to end_values over a set number of steps."""

    def __init__(
        self,
        *curriculum_args,
        start_values: List[SupportsFloat],
        end_values: List[SupportsFloat],
        total_steps: Union[int, List[int]],
        **curriculum_kwargs,
    ):
        """ Initialize the SimulatedAnnealing curriculum.

        :param start_values: The initial values of the curriculum.
        :param end_values: The final values of the curriculum.
        :param total_steps: The total number of steps to anneal the curriculum over.
        """
        super().__init__(*curriculum_args, **curriculum_kwargs)
        assert isinstance(
            self.task_space.gym_space, Box
        ), "AnnealingBoxCurriculum only supports Box task spaces."

        self.start_values = np.array(start_values, dtype=np.float32)
        self.end_values = np.array(end_values, dtype=np.float32)

        # Convert total_steps to list if necessary
        if isinstance(total_steps, SupportsInt):
            total_steps = [total_steps]
        self.total_steps = np.array(total_steps, dtype=np.int32)

        assert len(self.start_values) == len(self.end_values), "Length of start_values and end_values must be the same."
        assert all(x > 0 for x in self.total_steps), "All elements of total_steps must be greater than 0."

        self.current_step = 0

    def update_on_episode(self, episode_return, length, task, progress, env_id=None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        self.current_step += length

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        # Linear annealing from start_values to end_values
        annealed_values = (
            self.start_values + (self.end_values - self.start_values) *
            np.minimum(self.current_step, self.total_steps) / self.total_steps
        )

        return [annealed_values.copy() for _ in range(k)]
