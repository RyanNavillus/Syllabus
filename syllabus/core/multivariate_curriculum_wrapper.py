from typing import Any, List, Union

import numpy as np
from gymnasium.spaces import Dict, Tuple

from syllabus.core import CurriculumWrapper
from syllabus.task_space import TaskSpace


class MultitaskWrapper(CurriculumWrapper):
    """
    Uniform sampling for task spaces with multiple subspaces (Tuple or Dict)
    """
    # TODO: How do I use curriculum wrappers with the make_curriculum functions?

    def __init__(self, *args, num_components: int, component_names: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_components = num_components

        # Duplicate task space for each component
        if num_components is not None:
            self.task_space = TaskSpace(Tuple([self.task_space.gym_space for _ in range(
                num_components)]), (tuple(self.task_space.tasks),) * num_components)
        elif component_names is not None:
            self.task_space = TaskSpace(Dict({name: self.task_space.gym_space for name in component_names}), {
                                        name: self.task_space.tasks for name in component_names})

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        if isinstance(self.task_space.gym_space, Tuple):
            multivariate_dists = [self.curriculum._sample_distribution() for _ in self.task_space.gym_space.spaces]
        elif isinstance(self.task_space.gym_space, Dict):
            multivariate_dists = {name: self.curriculum._sample_distribution()
                                  for name in self.task_space.gym_space.keys()}
        else:
            raise NotImplementedError("Multivariate task space must be Tuple or Dict.")
        return multivariate_dists

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        assert self.num_tasks > 0, "Task space is empty. Please add tasks to the curriculum before sampling."

        tasks = []
        for _ in range(k):
            sample_dist = self._sample_distribution()
            if isinstance(sample_dist, list):
                task_components = []
                for dist in sample_dist:
                    task_components.append(self.curriculum.sample(k=1)[0])
                tasks.append(tuple(task_components))
        return tasks

        multitask_dist = self._sample_distribution()
        # TODO: Clean and comment
        if isinstance(self.task_space.gym_space, Dict):
            multitasks = []
            for _ in range(k):
                multitask = {}
                # TODO: Provide easier access to gym_space properties?
                for (space_name, task_space), task_dist in zip(self.task_space.tasks.items(), multitask_dist):
                    n_tasks = len(task_dist)
                    task_idx = np.random.choice(list(range(n_tasks)), size=1, p=task_dist)
                    multitask[space_name] = np.array([self.get_tasks(task_space)[i] for i in task_idx])
                multitasks.append(multitask)
            return multitasks
        elif isinstance(self.task_space.gym_space, Tuple):
            multitask = []
            for tasks, task_dist in zip(self.task_space.tasks, multitask_dist):
                print(tasks)
                n_tasks = len(task_dist)
                task_idx = np.random.choice(list(range(n_tasks)), size=k, p=task_dist)
                multitask.append(np.array([tasks[i] for i in task_idx]))
            multitask = np.array(multitask)
            return np.moveaxis(multitask, -1, 0)
        else:
            raise NotImplementedError("Multivariate task space must be Tuple or Dict.")

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        raise NotImplementedError("Multitask curriculum does not support logging metrics.")
