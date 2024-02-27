# flake8: noqa: F401
from typing import TypeVar

import gym
import gym_multi_car_racing
import numpy as np
from gym.spaces import Box
from tqdm.auto import tqdm

from syllabus.core import TaskWrapper

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")


class MultiCarRacingParallelWrapper(TaskWrapper):
    """
    Wrapper ensuring compatibility with the PettingZoo Parallel API.
    """

    def __init__(self, n_agents, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents

    def _actions_pz_to_np(self, action: dict[AgentID, ActionType]) -> np.ndarray:
        """
        Converts actions defined in PZ format to a numpy array.
        """
        assert action.__len__() == self.n_agents

        action = np.array(list(action.values()))
        assert action.shape == (self.n_agents, 3)
        return action

    def reset(self):
        pass

    def step(self, action: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """
        Takes inputs in the PettingZoo (PZ) Parallel API format, performs a step and
        returns outputs in PZ format.
        """

        def _np_array_to_pz_dict(array: np.ndarray):
            """
            Converts an n-dimensional numpy array to a dictionary
            with n keys and values.
            """
            out = {}
            for idx, i in enumerate(array):
                out[idx] = i
            return out

        def _singleton_to_pz_dict(value: bool):
            """
            Converts a boolean flag to a dictionary with ``n_agents`` keys and values.
            """
            return {idx: value for idx in range(self.n_agents)}

        # convert action to numpy format to perform the env step
        np_action = self._actions_pz_to_np(action)
        obs, rew, term, info = self.env.step(np_action)
        trunc = 0  # there is no `truncated` flag in this environment
        self.task_completion = self._task_completion(obs, rew, term, trunc, info)
        info["task_completion"] = self.task_completion
        # convert outputs back to PZ format
        obs, rew = tuple(map(_np_array_to_pz_dict, [obs, rew]))
        # TODO: are boolean flags replicated `n_agent` times in PZ format ?
        term, trunc = tuple(map(_singleton_to_pz_dict, [term, trunc]))

        return self.observation(obs), rew, term, trunc, info


if __name__ == "__main__":
    n_agents = 2
    env = gym.make(
        "MultiCarRacing-v0",
        num_agents=n_agents,
        direction="CCW",
        use_random_direction=True,
        backwards_flag=True,
        h_ratio=0.25,
        use_ego_color=False,
    )

    obs = env.reset()
    env = MultiCarRacingParallelWrapper(env=env, n_agents=n_agents)
    done = {i: False for i in range(n_agents)}
    total_reward = {i: 0 for i in range(n_agents)}
    np.random.seed(0)

    # while not all(done.values()):
    for steps in tqdm(range(100)):
        # The actions have to be of the format (num_agents,3)
        # The action format for each car is as in the CarRacing-v0 environment
        # i.e. (`Box([-1. 0. 0.], 1.0, (3,), float32)`)
        action = np.random.normal(0, 1, (2, 3))

        assert action.shape == (n_agents, 3)
        pz_action = {i: action[i] for i in range(n_agents)}
        # Similarly, the structure of this is the same as in CarRacing-v0 with an
        # additional dimension for the different agents, i.e.
        # obs is of shape (num_agents, 96, 96, 3)
        # reward is of shape (num_agents,)
        # done is a bool and info is not used (an empty dict).
        obs, reward, done, trunc, info = env.step(pz_action)
        for agent in range(n_agents):
            total_reward[agent] += reward[agent]
        env.render()

    print("individual scores:", total_reward)
    print(reward, done, trunc, info)
