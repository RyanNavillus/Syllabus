import typing
from typing import Any, List, Union

from syllabus.curricula import SequentialCurriculum


class SyncTestCurriculum(SequentialCurriculum):
    """
    Base class and API for defining curricula to interface with Gym environments.
    """
    REQUIRES_STEP_UPDATES = False
    REQUIRES_EPISODE_UPDATES = True
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, num_envs, num_episodes, *curriculum_args, **curriculum_kwargs):
        # Create a manual curriculum with a new task per episode, repeated across all envs
        task_list = [f"task {i+1}" for i in range(num_episodes)]
        stopping = [f"tasks>={num_envs}"] * (num_episodes - 1)
        super().__init__(task_list, stopping, *curriculum_args, **curriculum_kwargs)
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.task_counts = {self.task_space.encode(task): 0 for task in task_list}
        self.task_counts[0] = 0     # Error task
        self.total_reward = 0
        self.total_dones = 0
        self.metadata = {}

    def update_on_episode(self, episode_return, episode_len, episode_task, env_id: int = None) -> None:
        super().update_on_episode(episode_return, episode_len, episode_task, env_id)
        self.total_reward += episode_return
        self.total_dones += 1
        self.task_counts[episode_task] += 1

    def get_stats(self):
        return {
            "total_reward": self.total_reward,
            "total_dones": self.total_dones,
            "task_counts": self.task_counts
        }

    def sample(self, k: int = 1) -> Union[List, Any]:
        remaining_tasks = (self.num_episodes * self.num_envs) - self.total_tasks
        if remaining_tasks < k:
            tasks = super().sample(k=remaining_tasks) + [0] * (k - remaining_tasks)
        else:
            tasks = []
            while k > self.num_envs - self.n_tasks:
                tasks += super().sample(k=self.num_envs - self.n_tasks)
                k -= (self.num_envs - self.n_tasks)
            if k > 0:
                tasks += super().sample(k=k)
        return tasks
