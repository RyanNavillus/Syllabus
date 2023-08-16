import numpy as np
from gym.spaces import Tuple, Dict
from typing import Any, List, Union
from syllabus.core import Curriculum


class DomainRandomization(Curriculum):
    """A simple but strong baseline for curriculum learning that uniformly samples a task from the task space.
    """
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        return [1.0 / self.num_tasks for _ in range(self.num_tasks)]
    
    def add_task(self, task: Any) -> None:
        self.task_space.add_task(task)

