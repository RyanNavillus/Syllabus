import numpy as np
from gym.spaces import Tuple, Dict
from typing import Any, List, Union
from syllabus.core import Curriculum, increment_task_space


class Uniform(Curriculum):
    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        return [1.0 / self.n_tasks for _ in range(self.n_tasks)]
    
    def add_task(self, task: tuple) -> None:
        self.task_space = increment_task_space(self.task_space)

