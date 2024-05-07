import os
import time
from copy import deepcopy
from typing import TypeVar

import joblib
import numpy as np
from gymnasium import spaces
from scipy.special import softmax

from syllabus.core import Curriculum  # noqa: E402
from syllabus.task_space import TaskSpace  # noqa: E402

AgentType = TypeVar("AgentType")


class SelfPlay(Curriculum):
    def __init__(
        self,
        agent: AgentType,
        device: str,
        storage_path=None,  # unused
        max_agents=None,  # unused
        seed: int = 0,
    ):
        self.name = "SP"
        self.device = device
        self.agent = deepcopy(agent).to(self.device)
        self.task_space = TaskSpace(
            spaces.Discrete(1)
        )  # SelfPlay can only return agent_id = 0
        self.history = {
            "winrate": 0,
            "n_games": 0,
        }

    def update_agent(self, agent: AgentType) -> AgentType:
        self.agent = deepcopy(agent).to(self.device)

    def get_opponent(self, agent_id: int) -> AgentType:
        if agent_id is None:
            agent_id = 0
        assert agent_id == 0, (
            f"Self play only tracks the current agent."
            f"Expected agent id 0, got {agent_id}"
        )
        return self.agent

    def sample(self, k=1):
        return 0

    def update_winrate(self, opponent_id: int, opponent_reward: int) -> None:
        """
        Uses an incremental mean to update the opponent's winrate.
        """
        opponent_reward = opponent_reward > 0  # converts the reward to 0 or 1
        self.history["n_games"] += 1
        old_winrate = self.history["winrate"]
        n = self.history["n_games"]

        self.history["winrate"] = old_winrate + (opponent_reward - old_winrate) / n


class FictitiousSelfPlay(Curriculum):
    def __init__(
        self,
        agent: AgentType,
        device: str,
        storage_path: str,
        max_agents: int,
        seed: int = 0,
    ):
        self.name = "FSP"
        self.uid = int(time.time())
        self.device = device
        self.storage_path = storage_path
        self.seed = seed
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        self.current_agent_index = 0
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

    def update_agent(self, agent):
        """
        Saves the current agent instance to a pickle file.
        When the `max_agents` limit is met, older agent checkpoints are overwritten.
        """
        joblib.dump(
            agent,
            filename=(
                f"{self.storage_path}/{self.name}_{self.seed}_agent_checkpoint_"
                f"{self.current_agent_index % self.max_agents}.pkl"
            ),
        )
        if self.current_agent_index < self.max_agents:
            self.current_agent_index += 1

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

    def get_opponent(self, agent_id: int) -> AgentType:
        """Loads an agent from the buffer of saved agents."""
        return joblib.load(
            f"{self.storage_path}/{self.name}_{self.seed}_agent_checkpoint_{agent_id}.pkl"
        ).to(self.device)

    def sample(self, k=1):
        return np.random.randint(self.current_agent_index)


class PrioritizedFictitiousSelfPlay(Curriculum):
    def __init__(
        self,
        agent: AgentType,
        device: str,
        storage_path: str,
        max_agents: int,
        seed: int = 0,
    ):
        self.name = "PFSP"
        self.uid = int(time.time())
        self.device = device
        self.storage_path = storage_path
        self.seed = seed
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        self.current_agent_index = 0
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
        joblib.dump(
            agent,
            filename=(
                f"{self.storage_path}/{self.name}_{self.seed}_agent_checkpoint_"
                f"{self.current_agent_index % self.max_agents}.pkl"
            ),
        )
        if self.current_agent_index < self.max_agents:
            self.current_agent_index += 1

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

    def get_opponent(self, agent_id: int) -> AgentType:
        """
        Samples an agent id from the softmax distribution induced by winrates
        then loads the selected agent from the buffer of saved agents.
        """
        return joblib.load(
            f"{self.storage_path}/{self.name}_{self.seed}_agent_checkpoint_{agent_id}.pkl"
        ).to(self.device)

    def sample(self, k=1):
        logits = [
            self.history[agent_id]["winrate"]
            for agent_id in range(self.current_agent_index)
        ]
        return np.random.choice(
            np.arange(self.current_agent_index),
            p=softmax(logits),
        )
