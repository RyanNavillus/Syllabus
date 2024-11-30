import warnings
from copy import copy

import gymnasium as gym

from syllabus.core import PettingZooTaskEnv, TaskEnv
from syllabus.task_space import DiscreteTaskSpace


class SyncTestEnv(TaskEnv):
    def __init__(self, num_episodes, num_steps=100):
        super().__init__()
        self.num_steps = num_steps
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(self.num_steps), gym.spaces.Discrete(2)))
        self.task_space = DiscreteTaskSpace(gym.spaces.Discrete(num_episodes + 1),
                                            ["error task"] + [f"task {i+1}" for i in range(num_episodes)])
        self.task = "error_task"

    def reset(self, new_task=None):
        if new_task == "error task":
            warnings.warn("Received error task. This likely means that too many tasks are being requested.", stacklevel=2)
        if new_task is None:
            warnings.warn("No task provided. Resetting to error task.", stacklevel=2)
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


class PettingZooSyncTestEnv(PettingZooTaskEnv):
    def __init__(self, num_episodes, num_steps=100):
        super().__init__()
        self.num_steps = num_steps
        self.possible_agents = ["agent1", "agent2"]
        self._action_spaces = {agent: gym.spaces.Discrete(2) for agent in self.possible_agents}
        self.observation_spaces = {agent: gym.spaces.Tuple((gym.spaces.Discrete(self.num_steps), gym.spaces.Discrete(2)))
                                   for agent in self.possible_agents}
        self.task_space = DiscreteTaskSpace(gym.spaces.Discrete(num_episodes + 1),
                                            ["error task"] + [f"task {i+1}" for i in range(num_episodes)])
        self.task = "error_task"
        self.metadata = {"render.modes": ["human"]}

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, new_task=None):
        self.agents = copy(self.possible_agents)
        if new_task == "error task":
            print(ValueError("Received error task. This likely means that too many tasks are being requested."))
        self.task = new_task
        self._turn = 0
        obs = {agent: 0.5 for agent in self.agents}
        info = {agent: {"content": "reset", "task": self.task} for agent in self.agents}
        return obs, info

    def step(self, action):
        self._turn += 1

        obs = {agent: self.observation((self._turn, action[agent])) for agent in self.agents}
        rew = {agent: 1 for agent in self.agents}
        term = {agent: self._turn >= self.num_steps for agent in self.agents}
        trunc = {agent: False for agent in self.agents}
        info = {agent: {"content": "step", "task_completion": self._task_completion(obs, rew, all(term.values()), all(trunc.values()), {})}
                for agent in self.agents}
        if all(term.values()) or all(trunc.values()):
            self.agents = []
        return obs, rew, term, trunc, info
