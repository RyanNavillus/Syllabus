import warnings
import gymnasium as gym
from syllabus.core import TaskEnv
from syllabus.task_space import TaskSpace


class SyncTestEnv(TaskEnv):
    def __init__(self, num_episodes, num_steps=100):
        super().__init__()
        self.num_steps = num_steps
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(self.num_steps), gym.spaces.Discrete(2)))
        self.task_space = TaskSpace(gym.spaces.Discrete(num_episodes + 1), ["error task"] + [f"task {i+1}" for i in range(num_episodes)])
        self.task = "error_task"

    def reset(self, new_task=None):
        if new_task == "error task":
            warnings.warn("Received error task. This likely means that too many tasks are being requested.")
        if new_task is None:
            warnings.warn("No task provided. Resetting to error task.")
        self.task = new_task
        self._turn = 0
        return (self._turn, None), {"content": "reset", "task": self.task}

    def step(self, action):
        self._turn += 1

        obs = self.observation((self._turn, action))
        rew = 1
        term = self._turn >= self.num_steps
        trunc = False
        info = {"content": "step", "task_completion": self._task_completion(obs, rew, term, trunc, {})}
        return obs, rew, term, trunc, info
