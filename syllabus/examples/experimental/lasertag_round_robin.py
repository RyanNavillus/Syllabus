import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
from lasertag_dr import batchify, unbatchify
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from syllabus.core import TaskWrapper

sys.path.append("../../..")
from lasertag import (  # noqa
    LasertagArena1,
    LasertagArena2,
    LasertagCorridor1,
    LasertagCorridor2,
    LasertagCross,
    LasertagLargeCorridor,
    LasertagMaze1,
    LasertagMaze2,
    LasertagRuins,
    LasertagRuins2,
    LasertagSixteenRoomsN2,
    LasertagStar,
)

AgentType = TypeVar("AgentType")
AgentID = TypeVar("AgentID")
AgentCurriculum = TypeVar("AgentCurriculum")
EnvCurriculum = TypeVar("EnvCurriculum")
ActionType = TypeVar("ActionType")
ObsType = TypeVar("ObsType")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=bool, default=False)
    parser.add_argument(
        "--agent-curriculum-1", type=str, default="SP", choices=["SP", "FSP", "PFSP"]
    )
    parser.add_argument(
        "--env-curriculum-1", type=str, default="DR", choices=["DR", "PLR"]
    )
    parser.add_argument(
        "--agent-curriculum-2", type=str, default="SP", choices=["SP", "FSP", "PFSP"]
    )
    parser.add_argument(
        "--env-curriculum-2", type=str, default="DR", choices=["DR", "PLR"]
    )
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="lasertag_RR",
        help="the name of this experiment",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="syllabus",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="rpegoud",
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=".",
        help="the base directory for logging and wandb storage.",
    )

    args = parser.parse_args()
    return args


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(3 * 5 * 5, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x, flatten_start_dim=1):
        x = torch.flatten(x, start_dim=flatten_start_dim)
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None, flatten_start_dim=1):
        x = torch.flatten(x, start_dim=flatten_start_dim)
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class LasertagFixedParallelWrapper(TaskWrapper):
    """
    Wrapper ensuring compatibility with the PettingZoo Parallel API.
    Used with fixed Lasertag environments (deterministic resets for benchmarking).

    Lasertag Environment:
        * Action shape:  `n_agents` * `Discrete(5)`
        * Observation shape: Dict('image': Box(0, 255, (`n_agents`, 3, 5, 5), uint8))
    """

    def __init__(self, n_agents, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_agents = n_agents
        self.task = None
        self.episode_return = 0
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.n_steps = 0
        self.max_steps = self.env.max_steps

    def __getattr__(self, name):
        """
        Delegate attribute lookup to the wrapped environment if the attribute
        is not found in the LasertagParallelWrapper instance.
        """
        return getattr(self.env, name)

    def _np_array_to_pz_dict(self, array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing individual observations for each agent.
        Assumes that the batch dimension represents individual agents.
        """
        out = {}
        for idx, value in enumerate(array):
            out[self.possible_agents[idx]] = value
        return out

    def _singleton_to_pz_dict(self, value: bool) -> Dict[str, bool]:
        """
        Broadcasts the `done` and `trunc` flags to dictionaries keyed by agent id.
        """
        return {str(agent_index): value for agent_index in range(self.n_agents)}

    def reset(self) -> Tuple[Dict[AgentID, ObsType], Dict[AgentID, dict]]:
        """
        Resets the environment and returns a dictionary of observations
        keyed by agent ID.
        """
        obs = self.env.reset()  # random level generation
        pz_obs = self._np_array_to_pz_dict(obs["image"])

        return pz_obs

    def step(self, action: Dict[AgentID, ActionType], device: str) -> Tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, float],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
        Dict[AgentID, dict],
    ]:
        """
        Takes inputs in the PettingZoo (PZ) Parallel API format, performs a step and
        returns outputs in PZ format.
        """
        action = batchify(action, device)
        obs, rew, done, info = self.env.step(action)
        obs = obs["image"]
        trunc = False  # there is no `truncated` flag in this environment
        self.task_completion = self._task_completion(obs, rew, done, trunc, info)
        # convert outputs back to PZ format
        obs, rew = map(self._np_array_to_pz_dict, [obs, rew])
        done, trunc, info = map(
            self._singleton_to_pz_dict, [done, trunc, self.task_completion]
        )
        self.n_steps += 1

        return self.observation(obs), rew, done, trunc, info


@dataclass
class AgentConfig:
    agent_curriculum: str
    env_curriculum: str

    def __str__(self) -> str:
        return f"{self.env_curriculum}_{self.agent_curriculum}"


def load_agent(
    agent_config: AgentConfig,
    step: int,
    seed: int,
    device: str = "cpu",
) -> AgentType:

    return joblib.load(
        (
            f"lasertag_{str(agent_config)}_checkpoints/"
            f"{str(agent_config)}_{step}_seed_{seed}.pkl"
        )
    ).to(device)


def play_n_episodes(
    agent_1_config: Dict[str, Union[AgentCurriculum, EnvCurriculum]],
    agent_2_config: Dict[str, Union[AgentCurriculum, EnvCurriculum]],
    step: int,
    seed: int,
    n_episodes: int = 10,
    environment_id: str = "LasertagArena1",
) -> Dict[str, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = test_envs[environment_id]()  # 2 agents by default
    env = LasertagFixedParallelWrapper(env=env, n_agents=2)

    agent_1 = load_agent(agent_config=agent_1_config, step=step, seed=seed)
    agent_2 = load_agent(agent_config=agent_2_config, step=step, seed=seed)

    stack_size = 3
    frame_size = (env.agent_view_size, env.agent_view_size)
    max_cycles = env.max_steps
    n_agents = 2
    agent_c_rew, opp_c_rew = 0, 0

    """ALGO LOGIC: EPISODE STORAGE"""
    rb_obs = torch.zeros((max_cycles, n_agents, stack_size, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, n_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, n_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, n_agents)).to(device)

    """ TRAINING LOGIC """
    for episode in tqdm(range(n_episodes)):
        # collect an episode
        with torch.no_grad():

            next_obs = env.reset()

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                joint_obs = batchify(next_obs, device).squeeze()
                agent_obs, opponent_obs = joint_obs

                # get action from the agent and the opponent
                agent_1_action, *_ = agent_1.get_action_and_value(
                    agent_obs, flatten_start_dim=0
                )
                agent_2_action, *_ = agent_2.get_action_and_value(
                    opponent_obs, flatten_start_dim=0
                )
                # execute the environment and log data
                joint_actions = torch.tensor((agent_1_action, agent_2_action))
                next_obs, rewards, terms, truncs, info = env.step(
                    unbatchify(joint_actions, env.possible_agents), device
                )

                agent_c_rew += rewards["agent_0"]
                opp_c_rew += rewards["agent_1"]

                # add to episode storage
                rb_obs[step] = batchify(next_obs, device)
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = joint_actions

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    break

    return {
        f"agent_1_step_{step}_seed_{seed}_rewards": agent_c_rew / n_episodes,
        f"agent_2_step_{step}_seed_{seed}_rewards": opp_c_rew / n_episodes,
    }


test_envs = {
    "LasertagArena1": LasertagArena1,
    "LasertagArena2": LasertagArena2,
    "LasertagCorridor1": LasertagCorridor1,
    "LasertagCorridor2": LasertagCorridor2,
    "LasertagMaze1": LasertagMaze1,
    "LasertagMaze2": LasertagMaze2,
    "LasertagRuins": LasertagRuins,
    "LasertagRuins2": LasertagRuins2,
    "LasertagStar": LasertagStar,
    "LasertagCross": LasertagCross,
    "LasertagLargeCorridor": LasertagLargeCorridor,
    "LasertagSixteenRoomsN2": LasertagSixteenRoomsN2,
}


if __name__ == "__main__":
    args = parse_args()
    agent_1_config = AgentConfig(args.agent_curriculum_1, args.env_curriculum_1)
    agent_2_config = AgentConfig(args.agent_curriculum_2, args.env_curriculum_2)

    if not os.path.exists(f"{args.logging_dir}"):
        os.makedirs(f"{args.logging_dir}", exist_ok=True)

    logs = {}

    for checkpoint in [2000, 4000, 6000]:
        for seed in [1]:
            returns = play_n_episodes(
                agent_1_config=agent_1_config,
                agent_2_config=agent_2_config,
                step=checkpoint,
                seed=seed,
                n_episodes=args.n_episodes,
                environment_id="LasertagArena1",
            )
            print(returns)
            title = (
                f"{str(agent_1_config)}_VS_{str(agent_2_config)}_"
                f"checkpoint_{checkpoint}_seed_{seed}"
            )
            logs[title] = returns

    print(logs)
