
import json
import numpy as np

from typing import Dict, List
from syllabus.curricula import LearningProgress


class OMNI(LearningProgress):
    def __init__(self, *args, interestingness: Dict = None, **kwargs):

        super().__init__(*args, **kwargs)
        # Mapping from evaluated task to interestingness dictionary
        # Interestingness dictionary maps each task in the task space to a true/false rating
        # of interestingness, given that the agent has a high success rate at the evaluated task.
        # Each entry can be interepretes as the conditional interestingess of a task given proficiency in another task.
        # The default value assigns False to the evaluated task and True to all other tasks
        if interestingness is None:
            self.post_task_interestingness = {
                evaluated_task: {
                    task: task != evaluated_task
                    for task in self.tasks
                }
                for evaluated_task in self.tasks
            }
        else:
            self.post_task_interestingness = interestingness

    def set_interestingess(self, interestingness: Dict) -> None:
        """Set the interestingness dictionary for the curriculum.

        :param interestingness: Dictionary mapping evaluated tasks to interestingness dictionaries
        """
        self.post_task_interestingness = interestingness

    def _sample_distribution(self) -> List[float]:
        # Prioritize tasks by learning progress first
        lp_dist = super()._sample_distribution()

        interesting_tasks = set()
        boring_tasks = set()
        # Sort tasks by success rate
        tasks_by_success = np.argsort(self.task_rates)

        # Iteratively add the most successful tasks to the interesting set.
        for task_idx in tasks_by_success[::-1]:
            if task_idx not in interesting_tasks and task_idx not in boring_tasks:
                interesting_tasks.add(task_idx)

                # Add tasks that are boring given proficiency at the successful task to the boring set.
                for task_idx, task in enumerate(self.tasks):
                    if (task not in self.post_task_interestingness
                        or (task_idx not in interesting_tasks and task_idx not in boring_tasks
                            and not self.post_task_interestingness[self.tasks[task_idx]][task])):
                        boring_tasks.add(task_idx)

        # Scale LP sampling probabilities by 1.0 for interesting tasks, 0.001 for boring tasks.
        moi_weight = np.ones(len(lp_dist))
        for i in boring_tasks:
            moi_weight[i] = 0.001

        # Scale and normalize
        omni_dist = lp_dist * moi_weight
        omni_dist = omni_dist / np.sum(omni_dist)
        return omni_dist


def interestingness_from_json(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        interestingness = json.load(f)
    return interestingness
