import typing
import numpy as np
import itertools
from gym.spaces import Tuple, Dict
from typing import Any, Callable, List, Union
from syllabus.core import Curriculum

class Uniform(Curriculum):
    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        return [1.0 / self.n_tasks for _ in range(self.n_tasks)]
    

class MultitaskUniform(Curriculum):
    """
    Uniform sampling for task spaces with multiple subspaces (Tuple or Dict)
    """
    def __init__(self, num_teams: int, *curriculum_args, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self.num_teams = num_teams

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        multivariate_dists = []
        if isinstance(self.task_space, Tuple):
            for space in self.task_space.spaces:
                n_tasks = self._n_tasks(space)
                multivariate_dists.append([1.0 / n_tasks for _ in range(n_tasks)])
        return multivariate_dists
    
    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        assert self.n_tasks > 0, "Task space is empty. Please add tasks to the curriculum before sampling."

        multitask_dist = self._sample_distribution()

        if isinstance(self.task_space, Tuple):
            multitask = []
            for task_space, task_dist in zip(self.task_space.spaces, multitask_dist):
                n_tasks = len(task_dist)
                task_idx = np.random.choice(list(range(n_tasks)), size=k, p=task_dist)
                multitask.append(np.array([self._tasks(task_space)[i] for i in task_idx]))
        elif isinstance(self.task_space, Dict):
            multitask = {}
        
        # Should be n_teams * 1 * k
        # Want k * n_teams * 1
        # Got n_teams * n_teams * n_teams
        multitask = np.array(multitask)
        return np.moveaxis(multitask, -1, 0)