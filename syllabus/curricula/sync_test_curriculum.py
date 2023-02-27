import typing
import numpy as np
from typing import Any, Dict, List, Union
from syllabus.core import Curriculum


class SyncTestCurriculum(Curriculum):
    """
    Base class and API for defining curricula to interface with Gym environments.
    """
    def __init__(self, *curriculum_args, **curriculum_kwargs):
        super().__init__(*curriculum_args, **curriculum_kwargs)
        self.expected_tasks = []
        self.total_reward = 0
        self.total_dones = 0

    def _complete_task(self, task: typing.Any, success_prob: float) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        try:
            self.expected_tasks.remove(task)
        except ValueError:
            raise ValueError("Recieved unexpected task {}".format(task))

    def _on_step(self, obs, rew, done, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        self.total_reward += rew
        self.total_dones += 1

    def _on_step_batch(self, step_results: List[typing.Tuple[int, int, int, int]]) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        for step_result in step_results:
            self.on_step(*step_result)

    def _on_episode(self, episode_return: float, trajectory: List = None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        raise NotImplementedError("Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def _on_demand(metrics: Dict):
        """
        Update the curriculum with arbitrary inputs.
        """
        raise NotImplementedError

    def batch_update_curriculum(self, update_data: List[Dict]):
        """
        Update the curriculum with the specified update type.
        """
        for update in update_data:
            self.update_curriculum(update)

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        # Uniform distribution
        return [1.0 / self._n_tasks for _ in range(self._n_tasks)]

    def sample(self, k: int = 1) -> Union[List, Any]:
        """
        Sample k tasks from the curriculum.
        """
        task_dist = self._sample_distribution()

        # Use list of indices because np.choice does not play nice with tuple tasks
        tasks = self._tasks
        n_tasks = len(tasks)
        task_idx = np.random.choice(list(range(n_tasks)), size=k, p=task_dist)
        tasks = [tasks[i] for i in task_idx]
        self.expected_tasks += tasks
        if len(self.expected_tasks) > 100:
            raise ValueError("Too many unused tasks in queue.")
        return tasks
