# flake8: noqa: F401
from typing import TypeVar

import gym
import gym_multi_car_racing
import numpy as np
from gymnasium import spaces
from tqdm.auto import tqdm

from syllabus.core import TaskWrapper, make_multiprocessing_curriculum
from syllabus.curricula import DomainRandomization
from syllabus.task_space import TaskSpace

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")


class MultiCarRacingParallelWrapper(TaskWrapper):
    """
    Wrapper ensuring compatibility with the PettingZoo Parallel API.

    Car Racing Environment:
        * Action shape:  ``n_agents`` * `Box([-1. 0. 0.], 1.0, (3,), float32)`
        * Observation shape: ``n_agents`` * `Box(0, 255, (96, 96, 3), uint8)`
        * Done: ``done`` is a single boolean value
        * Info: ``info`` is unused and represented as an empty dictionary
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

    def _np_array_to_pz_dict(self, array: np.ndarray) -> dict[int : np.ndarray]:
        """
        Returns a dictionary containing individual observations for each agent.
        """
        out = {}
        for idx, i in enumerate(array):
            out[idx] = i
        return out

    def _singleton_to_pz_dict(self, value: bool) -> dict[int:bool]:
        """
        Broadcasts the `done` and `trunc` flags to dictionaries keyed by agent id.
        """
        return {idx: value for idx in range(self.n_agents)}

    def reset(self) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """
        Resets the environment and returns a dictionary of observations
        keyed by agent ID.
        """
        # TODO: what is the second output (dict[AgentID, dict]])?
        obs = self.env.reset()
        pz_obs = self._np_array_to_pz_dict(obs)

        return pz_obs

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
        # convert action to numpy format to perform the env step
        np_action = self._actions_pz_to_np(action)
        obs, rew, term, info = self.env.step(np_action)
        trunc = 0  # there is no `truncated` flag in this environment
        self.task_completion = self._task_completion(obs, rew, term, trunc, info)
        # convert outputs back to PZ format
        obs, rew = tuple(map(self._np_array_to_pz_dict, [obs, rew]))
        term, trunc, info = tuple(
            map(self._singleton_to_pz_dict, [term, trunc, self.task_completion])
        )

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

    env = MultiCarRacingParallelWrapper(env=env, n_agents=n_agents)
    # curriculum = DomainRandomization(env)
    # curriculum, task_queue, update_queue = make_multiprocessing_curriculum(
    #     curriculum,
    # )

    done = {i: False for i in range(n_agents)}
    total_reward = {i: 0 for i in range(n_agents)}
    np.random.seed(0)

    # while not all(done.values()):
    for episodes in tqdm(range(5)):  # testing with 5 truncated episodes
        obs = env.reset()
        for steps in range(100):
            action = np.random.normal(0, 1, (2, 3))
            pz_action = {i: action[i] for i in range(n_agents)}
            obs, reward, done, trunc, info = env.step(pz_action)
            for agent in range(n_agents):
                total_reward[agent] += reward[agent]
            env.render()

    print("individual scores:", total_reward)
    print(reward, done, trunc, info)
