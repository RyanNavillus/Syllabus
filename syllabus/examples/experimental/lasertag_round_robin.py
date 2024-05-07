import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar

import joblib
import numpy as np
import torch
import torch.nn as nn
from syllabus.examples.experimental.lasertag_dr import batchify, unbatchify
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-curriculum-1", type=str, default="SP", choices=["SP", "FSP", "PFSP"]
    )
    parser.add_argument(
        "--env-curriculum-1", type=str, default="DR", choices=["DR", "PLR"]
    )
    parser.add_argument(
        "--base-path-1", type=str, default="",
    )
    parser.add_argument(
        "--agent-curriculum-2", type=str, default="SP", choices=["SP", "FSP", "PFSP"]
    )
    parser.add_argument(
        "--env-curriculum-2", type=str, default="DR", choices=["DR", "PLR"]
    )
    parser.add_argument(
        "--base-path-2", type=str, default="",
    )
    parser.add_argument("--n-episodes", type=int, default=10)
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
    """
    Dataclass used to store information about the current version
    of an agent, as defined by its checkpoint, seed and curriculums.
    """

    agent_curriculum: str
    env_curriculum: str
    base_path: str
    agent = None
    checkpoint: int = None
    seed: int = None

    def __post_init__(self):
        self.get_checkpoints_and_seeds()

    def set_task(self, checkpoint: int, seed: int) -> None:
        self.checkpoint = checkpoint
        self.seed = seed

    @property
    def path(self) -> str:
        return f"{self.env_curriculum}_{self.agent_curriculum}"

    def get_checkpoints_and_seeds(self) -> None:
        """Extracts all unique checkpoints and seeds from a checkpoint folder path."""
        files = os.listdir(f"{self.base_path}")
        checkpoints = [file for file in files if file != "cached"]
        self.checkpoints = set(map(lambda x: x.split("_")[2], checkpoints))
        self.seeds = set(map(lambda x: x.split("_")[-1].split(".")[0], checkpoints))

    def __str__(self) -> str:
        return (
            f"{self.env_curriculum}_{self.agent_curriculum}_"
            f"{self.checkpoint}_seed_{self.seed}"
        )


def load_agent(
    agent_cfg: AgentConfig,
    device: str = "cpu",
) -> Tuple[AgentType, AgentConfig]:
    """Loads an agent to `device` and updates `AgentConfig`"""
    path = f"{agent_cfg.base_path}/{str(agent_cfg)}.pkl"
    try:
        agent = joblib.load(path)
        agent = agent.to(device)

        agent_cfg.agent = agent
    except FileNotFoundError:
        return None, agent_cfg

    return agent, agent_cfg


def play_n_episodes(
    agent_1_cfg: AgentConfig,
    agent_2_cfg: AgentConfig,
    device: str,
    n_episodes: int = 10,
    environment_id: str = "LasertagArena1",
) -> Tuple[float, float]:

    n_agents = 2
    stack_size = 3
    env = test_envs[environment_id]()  # 2 agents by default
    env = LasertagFixedParallelWrapper(env=env, n_agents=n_agents)

    frame_size = (env.agent_view_size, env.agent_view_size)
    max_cycles = env.max_steps
    agent_1_c_rew, agent_2_c_rew = 0, 0

    """ALGO LOGIC: EPISODE STORAGE"""
    rb_obs = torch.zeros((max_cycles, n_agents, stack_size, *frame_size)).to(device)
    rb_actions = torch.zeros((max_cycles, n_agents)).to(device)
    rb_rewards = torch.zeros((max_cycles, n_agents)).to(device)
    rb_terms = torch.zeros((max_cycles, n_agents)).to(device)

    agent_1, agent_2 = agent_1_cfg.agent, agent_2_cfg.agent

    """ TRAINING LOGIC """
    for episode in range(n_episodes):
        # collect an episode
        with torch.no_grad():

            next_obs = env.reset()

            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                joint_obs = batchify(next_obs, device).squeeze()
                agent_obs, opponent_obs = joint_obs.split(1, dim=0)

                agent_obs = agent_obs.squeeze().to(device)
                opponent_obs = opponent_obs.squeeze().to(device)

                # get action from the agent and the opponent
                agent_1_action, *_ = agent_1.get_action_and_value(
                    agent_obs, flatten_start_dim=0
                )
                agent_2_action, *_ = agent_2.get_action_and_value(
                    opponent_obs, flatten_start_dim=0
                )
                # execute the environment and log data
                joint_actions = torch.tensor((agent_1_action, agent_2_action))
                next_obs, rewards, terms, truncs, _ = env.step(
                    unbatchify(joint_actions, env.possible_agents), device
                )

                agent_1_c_rew += rewards["agent_0"]
                agent_2_c_rew += rewards["agent_1"]

                # add to episode storage
                rb_obs[step] = batchify(next_obs, device)
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = joint_actions

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    break

    agent_1_norm_rew = agent_1_c_rew / n_episodes
    agent_2_norm_rew = agent_2_c_rew / n_episodes

    return agent_1_norm_rew, agent_2_norm_rew


if __name__ == "__main__":
    args = parse_args()
    agent_1_cfg = AgentConfig(args.agent_curriculum_1, args.env_curriculum_1, args.base_path_1)
    agent_2_cfg = AgentConfig(args.agent_curriculum_2, args.env_curriculum_2, args.base_path_2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(f"{args.logging_dir}"):
        os.makedirs(f"{args.logging_dir}", exist_ok=True)

    logs = {
        agent.path: {checkpoint: [] for checkpoint in agent.checkpoints}
        for agent in (agent_1_cfg, agent_2_cfg)
    }

    for checkpoint_1 in tqdm(agent_1_cfg.checkpoints, desc="outer loop"):
        for seed_1 in agent_1_cfg.seeds:
            agent_1_cfg.set_task(checkpoint_1, seed_1)
            agent_1, agent_1_cfg = load_agent(agent_1_cfg, device)

            for checkpoint_2 in tqdm(agent_2_cfg.checkpoints, desc="inner loop"):
                for seed_2 in agent_2_cfg.seeds:
                    agent_2_cfg.set_task(checkpoint_2, seed_2)
                    agent_2, agent_2_cfg = load_agent(agent_2_cfg, device)

                    if agent_1 is None or agent_2 is None:
                        print(f"Skipping checkpoints=[{checkpoint_1},{checkpoint_2}] seeds=[{seed_1},{seed_2}]")
                        continue

                    for environment_id in list(test_envs.keys()):
                        returns_1, returns_2 = play_n_episodes(
                            agent_1_cfg,
                            agent_2_cfg,
                            device,
                            n_episodes=args.n_episodes,
                            environment_id=environment_id,
                        )
                        logs[agent_1_cfg.path][checkpoint_1].append(returns_1)
                        logs[agent_2_cfg.path][checkpoint_2].append(returns_2)

    if not os.path.exists(f"{args.logging_dir}/round_robin/"):
        os.makedirs(f"{args.logging_dir}/round_robin/")

    with open(
        f"{args.logging_dir}/round_robin/{agent_1_cfg.path}_{agent_2_cfg.path}.json",
        "w",
    ) as outfile:
        json.dump(logs, outfile)
