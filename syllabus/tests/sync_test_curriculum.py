import typing
from typing import Any, List, Union

from syllabus.curricula import SequentialCurriculum


class SyncTestCurriculum(SequentialCurriculum):
    """
    Base class and API for defining curricula to interface with Gym environments.
    """
    REQUIRES_STEP_UPDATES = True
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, num_envs, num_episodes, *curriculum_args, **curriculum_kwargs):
        # Create a manual curriculum with a new task per episode, repeated across all envs
        task_list = [f"task {i+1}" for i in range(num_episodes)]
        num_repeats = [num_envs] * num_episodes
        super().__init__(task_list, *curriculum_args, num_repeats=num_repeats, repeat_list=False, **curriculum_kwargs)
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.task_counts = {task: 0 for task in task_list}
        self.task_counts["error task"] = 0
        self.total_reward = 0
        self.total_dones = 0

    def update_task_progress(self, task: typing.Any, progress: float) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        if progress > 0.999:
            self.task_counts[task] += 1

    def update_on_step(self, obs, rew, term, trunc, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        self.total_reward += rew
        if term or trunc:
            self.total_dones += 1

    def get_stats(self):
        return {
            "total_reward": self.total_reward,
            "total_dones": self.total_dones,
            "task_counts": self.task_counts
        }

    def sample(self, k: int = 1) -> Union[List, Any]:
        remaining_tasks = self.remaining_tasks()
        if remaining_tasks < k:
            tasks = super().sample(k=remaining_tasks) + ["error task"] * (k - remaining_tasks)
        else:
            tasks = super().sample(k=k)
        return tasks
