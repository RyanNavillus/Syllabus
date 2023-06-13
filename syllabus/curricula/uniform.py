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
        

