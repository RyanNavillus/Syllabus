import typing
import numpy as np
import itertools
from gym.spaces import Tuple, Dict
from typing import Any, Callable, List, Union
from syllabus.core import Curriculum, CurriculumWrapper


class MultitaskWrapper(CurriculumWrapper):
    """
    Uniform sampling for task spaces with multiple subspaces (Tuple or Dict)
    """
    # TODO: How do I use curriculum wrappers with the make_curriculum functions?
    def __init__(self, *args, num_components: int = None, component_names: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_components is not None or component_names is not None, "Must specify either num_components or component_names."
        if num_components is not None:
            self.task_space = Tuple([self.task_space for _ in range(num_components)])
        elif component_names is not None:
            self.task_space = Dict({name: self.task_space for name in component_names})

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        if isinstance(self.task_space, Tuple):
            multivariate_dists = [self.curriculum._sample_distribution() for _ in self.task_space.spaces]
        elif isinstance(self.task_space, Tuple):
            multivariate_dists = {name: self.curriculum._sample_distribution() for name in self.task_space.keys()}
        else:
            raise NotImplementedError("Multivariate task space must be Tuple or Dict.")
        return multivariate_dists
    
    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        assert self.n_tasks > 0, "Task space is empty. Please add tasks to the curriculum before sampling."

        multitask_dist = self._sample_distribution()

        if isinstance(self.task_space, Dict):
            multitasks = []
            for _ in range(k):
                multitask = {}
                for (space_name, task_space), task_dist in zip(self.task_space.spaces.items(), multitask_dist):
                    n_tasks = len(task_dist)
                    task_idx = np.random.choice(list(range(n_tasks)), size=1, p=task_dist)
                    multitask[space_name] = np.array([self._tasks(task_space)[i] for i in task_idx])
                multitasks.append(multitask)
            return multitasks
        elif isinstance(self.task_space, Tuple):
            multitask = []
            for task_space, task_dist in zip(self.task_space.spaces, multitask_dist):
                n_tasks = len(task_dist)
                task_idx = np.random.choice(list(range(n_tasks)), size=k, p=task_dist)
                multitask.append(np.array([self._tasks(task_space)[i] for i in task_idx]))
            multitask = np.array(multitask)
            return np.moveaxis(multitask, -1, 0)
        else:
            raise NotImplementedError("Multivariate task space must be Tuple or Dict.")    

    def log_metrics(self, writer, step=None):
        raise NotImplementedError("Multitask curriculum does not support logging metrics.")