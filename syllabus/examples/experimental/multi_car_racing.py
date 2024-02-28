# flake8: noqa: F401
from typing import TypeVar

import gym
import gym_multi_car_racing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions.normal import Normal
from tqdm.auto import tqdm

from syllabus.core import TaskWrapper, make_multiprocessing_curriculum
from syllabus.curricula import DomainRandomization
from syllabus.task_space import TaskSpace

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    x = np.stack([x[a] for a in x], axis=0)
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


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
        self.task = None
        self.episode_return = 0
        self.task_space = TaskSpace(
            spaces.Box(
                low=np.array([-1.0, 0.0, 0.0]),
                high=np.array([1.0, 1.0, 1.0]),
                shape=(3,),
                dtype=np.float32,
            )
        )

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
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    stack_size = 4
    frame_size = (96, 96)
    max_cycles = 125
    total_episodes = 100

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

    """ CURRICULUM SETUP """
    env = MultiCarRacingParallelWrapper(env=env, n_agents=n_agents)
    curriculum = DomainRandomization(env.task_space)
    curriculum, task_queue, update_queue = make_multiprocessing_curriculum(curriculum)

    # TODO: clarify how to setup continuous PPO
    # """ LEARNER SETUP """
    # agent = Agent(envs=envs).to(device)
    # optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

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
