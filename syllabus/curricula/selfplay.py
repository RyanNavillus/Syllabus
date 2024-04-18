import os
from copy import deepcopy
from typing import TypeVar

import joblib
import numpy as np
from gymnasium import spaces
from scipy.special import softmax

from syllabus.core import Curriculum  # noqa: E402
from syllabus.task_space import TaskSpace  # noqa: E402

Agent = TypeVar("Agent")


class SelfPlay(Curriculum):
    def __init__(
        self,
        agent: Agent,
        device: str,
        storage_path=None,  # unused
        max_agents=None,  # unused
    ):
        self.agent = deepcopy(agent).to(self.device)
        self.device = device
        self.task_space = TaskSpace(
            spaces.Discrete(1)
        )  # SelfPlay can only return agent_id = 0

    def update_agent(self, agent: Agent) -> Agent:
        self.agent = deepcopy(agent).to(self.device)

    def get_opponent(self, agent_id) -> Agent:
        if agent_id is None:
            agent_id = 0
        assert (
            agent_id == 0
        ), f"Self play only tracks the current agent. Expected agent id 0, got {agent_id}"
        return self.agent

    def sample(self, k=1):
        return 0


class FictitiousSelfPlay(Curriculum):
    def __init__(
        self,
        agent: Agent,
        device: str,
        storage_path: str,
        max_agents: int,
    ):
        self.name = "FSP"
        self.device = device
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        self.n_stored_agents = 0
        self.max_agents = max_agents
        self.task_space = TaskSpace(spaces.Discrete(self.max_agents))
        self.update_agent(agent)  # creates the initial opponent

    def update_agent(self, agent):
        """Saves the current agent instance to a pickle file."""
        if self.n_stored_agents < self.max_agents:
            # TODO: define the expected behaviour when the limit is exceeded
            joblib.dump(
                agent,
                filename=f"{self.storage_path}/{self.name}_agent_checkpoint_{self.n_stored_agents}.pkl",
            )
            self.n_stored_agents += 1

    def get_opponent(self, agent_id) -> Agent:
        """Loads an agent from the buffer of saved agents."""
        return joblib.load(
            f"{self.storage_path}/{self.name}_agent_checkpoint_{agent_id}.pkl"
        ).to(self.device)

    def sample(self, k=1):
        return np.random.randint(self.n_stored_agents)


class PrioritizedFictitiousSelfPlay(Curriculum):
    def __init__(
        self,
        agent: Agent,
        device: str,
        storage_path: str,
        max_agents: int,
    ):
        self.name = "PFSP"
        self.device = device
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        self.n_stored_agents = 0
        self.max_agents = max_agents
        self.task_space = TaskSpace(spaces.Discrete(self.max_agents))
        self.update_agent(agent)  # creates the initial opponent
        self.history = {
            i: {
                "winrate": 0,
                "n_games": 0,
            }
            for i in range(self.max_agents)
        }

    def update_agent(self, agent) -> None:
        """
        Saves the current agent instance to a pickle file and update
        its priority.
        """
        if self.n_stored_agents < self.max_agents:
            # TODO: define the expected behaviour when the limit is exceeded
            joblib.dump(
                agent,
                filename=f"{self.storage_path}/{self.name}_agent_checkpoint_{self.n_stored_agents}.pkl",
            )
            self.n_stored_agents += 1

    def update_winrate(self, opponent_id: int, opponent_reward: int) -> None:
        """
        Uses an incremental mean to update the opponent's winrate i.e. priority.
        This implies that sampling according to the winrates returns the most
        challenging opponents.
        """
        opponent_reward = opponent_reward > 0  # converts the reward to 0 or 1
        self.history[opponent_id]["n_games"] += 1
        old_winrate = self.history[opponent_id]["winrate"]
        n = self.history[opponent_id]["n_games"]

        self.history[opponent_id]["winrate"] = (
            old_winrate + (opponent_reward - old_winrate) / n
        )

    def get_opponent(self, agent_id: int) -> Agent:
        """
        Samples an agent id from the softmax distribution induced by winrates
        then loads the selected agent from the buffer of saved agents.
        """
        return joblib.load(
            f"{self.storage_path}/{self.name}_agent_checkpoint_{agent_id}.pkl"
        ).to(self.device)

    def sample(self, k=1):
        logits = [
            self.history[agent_id]["winrate"]
            for agent_id in range(self.n_stored_agents)
        ]
        return np.random.choice(
            np.arange(self.n_stored_agents),
            p=softmax(logits),
        )
