import gym

from syllabus.task_space import TaskSpace
from syllabus.core import TaskEnv

class SyncTestEnv(TaskEnv):
    def __init__(self, num_episodes):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(10), gym.spaces.Discrete(2)))
        self.task_space = TaskSpace(gym.spaces.Discrete(num_episodes+1), ["error task"] + [f"task {i+1}" for i in range(num_episodes)])
        self.task = "task 1"

    def reset(self, new_task=None):
        if new_task == "error task":
            print(ValueError("Received error task. This likely means that too many tasks are being requested."))
        self.task = new_task
        self._turn = 0
        return 0.5, 1, False, {"content": "reset", "task": self.task}
     
    def step(self, action):
        self._turn += 1

        obs = self.observation((self._turn, action))
        rew = 1
        done = self._turn >= 10
        info = {"content": "step", "task_completion": self._task_completion(obs, rew, done, {})}
        return obs, rew, done, info


