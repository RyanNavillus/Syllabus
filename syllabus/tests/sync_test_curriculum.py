from collections import defaultdict
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
        stopping = [f"tasks>={num_envs}"] * (num_episodes - 1)
        super().__init__(task_list, stopping, *curriculum_args, **curriculum_kwargs)
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.task_counts = {self.task_space.encode(task): 0 for task in task_list}
        self.task_counts[0] = 0     # Error task
        self.total_reward = 0
        self.total_dones = 0
        self.episode_rewards = defaultdict(int)
        self.metadata = {}

    @property
    def requires_step_updates(self) -> bool:
        return True

    def update_on_episode(self, episode_return, length, task, progress, env_id=None) -> None:
        super().update_on_episode(episode_return, length, task, progress, env_id)
        self.total_reward += episode_return
        self.total_dones += 1
        self.task_counts[task] += 1

        assert self.episode_rewards[
            env_id] == episode_return, f"Episode return {episode_return} does not match expected {self.episode_rewards[env_id]}"
        self.episode_rewards[env_id] = 0

    def update_on_step(self, task, obs, rew, term, trunc, info, progress, env_id=None) -> None:
        super().update_on_step(task, obs, rew, term, trunc, info, env_id)
        self.episode_rewards[env_id] += rew

    def update_on_step_batch(self, step_results, env_id=None):
        # print(step_results)
        tasks, obs, rews, terms, truncs, infos, progresses = tuple(step_results)
        for t, o, r, te, tr, i, p in zip(tasks, obs, rews, terms, truncs, infos, progresses):
            self.update_on_step(t, o, r, te, tr, i, p, env_id=env_id)

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
