""" Self play curricula for training agents against copies themselves. This is an experimental API and subject to change."""
import os
import time
from copy import deepcopy
from typing import List, TypeVar

import joblib
import numpy as np
from gymnasium import spaces
from scipy.special import softmax

from syllabus.core import Curriculum, Agent  # noqa: E402
from syllabus.task_space import TaskSpace  # noqa: E402


class SelfPlay(Curriculum):
    """Self play curriculum for training agents against themselves."""

    def __init__(
        self,
        task_space: TaskSpace,
        agent: Agent,
        device: str,
    ):
        """ Initialize the self play curriculum.

        :param task_space: The task space of the environment
        :param agent: The initial agent to play against
        :param device: The device to run the agent on
        """
        super().__init__(task_space)
        self.device = device
        self.agent = deepcopy(agent).to(self.device)
        self.task_space = TaskSpace(
            spaces.Discrete(1)
        )  # SelfPlay can only return agent_id = 0
        self.history = {
            "winrate": 0,
            "n_games": 0,
        }

    def add_agent(self, agent: Agent) -> int:
        self.agent = deepcopy(agent).to(self.device)
        return 0

    def get_agent(self, agent_id: int) -> Agent:
        if agent_id is None:
            agent_id = 0
        assert agent_id == 0, (
            f"Self play only tracks the current agent."
            f"Expected agent id 0, got {agent_id}"
        )
        return self.agent

    def _sample_distribution(self) -> List[float]:
        return [1.0]

    def sample(self, k=1):
        return [0 for _ in range(k)]

    def update_winrate(self, agent_id: int, reward: int) -> None:
        """
        Uses an incremental mean to update an agent's winrate. This assumes that reward
        is positive for a win and negative for a loss. Not used for sampling.

        :param agent_id: Identifier of the agent
        :param reward: Reward received by the agent
        """
        win = reward > 0  # converts the reward to 0 or 1
        self.history["n_games"] += 1
        old_winrate = self.history["winrate"]
        n = self.history["n_games"]

        self.history["winrate"] = old_winrate + (win - old_winrate) / n

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """Log metrics for the curriculum."""
        logs.append("winrate", self.history["winrate"])
        logs.append("n_games", self.history["n_games"])
        super().log_metrics(writer, logs, step, log_n_tasks)


class FictitiousSelfPlay(Curriculum):

    def __init__(
        self,
        task_space: TaskSpace,
        agent: Agent,
        device: str,
        storage_path: str,
        max_agents: int,
        seed: int = 0,
        max_loaded_agents: int = 1,
    ):
        super().__init__(task_space)
        self.uid = int(time.time())
        self.device = device
        self.storage_path = storage_path
        self.seed = seed

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        self.current_agent_index = 0
        self.max_agents = max_agents
        self.task_space = TaskSpace(spaces.Discrete(self.max_agents))
        self.add_agent(agent)  # creates the initial opponent
        self.history = {
            i: {
                "winrate": 0,
                "n_games": 0,
            }
            for i in range(self.max_agents)
        }
        self.loaded_agents = {i: None for i in range(self.max_agents)}
        self.n_loaded_agents = 0
        self.max_loaded_agents = max_loaded_agents

    def add_agent(self, agent):
        """
        Saves the current agent instance to a pickle file.
        When the `max_agents` limit is met, older agent checkpoints are overwritten.
        """
        agent = agent.to("cpu")
        joblib.dump(
            agent,
            filename=(
                f"{self.storage_path}/{self.name}_{self.seed}_agent_checkpoint_"
                f"{self.current_agent_index % self.max_agents}.pkl"
            ),
        )
        agent = agent.to(self.device)
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

    def get_agent(self, agent_id: int) -> Agent:
        """Loads an agent from the buffer of saved agents."""
        if self.loaded_agents[agent_id] is None:
            if self.n_loaded_agents >= self.max_loaded_agents:
                pass
            print(
                "get agent",
                agent_id,
                f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_{agent_id}.pkl",
            )
            self.loaded_agents[agent_id] = joblib.load(
                f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_{agent_id}.pkl"
            ).to(self.device)

        return self.loaded_agents[agent_id]

    def _sample_distribution(self) -> List[float]:
        return [1.0 / self.n_loaded_agents for _ in range(self.n_loaded_agents)] \
            + [0.0 for _ in range(self.max_agents - self.n_loaded_agents)]

    def sample(self, k=1):
        probs = self._sample_distribution()
        return list(np.random.choice(
            np.arange(self.current_agent_index),
            p=probs,
            size=k,
        ))

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """Log metrics for the curriculum."""
        logs.append("winrate", self.history["winrate"])
        logs.append("games_played", self.history["n_games"])
        logs.append("stored_agents", self.n_loaded_agents)
        super().log_metrics(writer, logs, step, log_n_tasks)


class PrioritizedFictitiousSelfPlay(Curriculum):

    def __init__(
        self,
        task_space: TaskSpace,
        agent: Agent,
        device: str,
        storage_path: str,
        max_agents: int,
        seed: int = 0,
        max_loaded_agents: int = 1,
    ):
        super().__init__(task_space)
        self.uid = int(time.time())
        self.device = device
        self.storage_path = storage_path
        self.seed = seed
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

        self.current_agent_index = 0
        self.max_agents = max_agents
        self.task_space = TaskSpace(spaces.Discrete(self.max_agents))
        self.add_agent(agent)  # creates the initial opponent
        self.history = {
            i: {
                "winrate": 0,
                "n_games": 0,
            }
            for i in range(self.max_agents)
        }
        self.loaded_agents = {i: None for i in range(self.max_agents)}
        self.n_loaded_agents = 0
        self.max_loaded_agents = max_loaded_agents

    def add_agent(self, agent) -> None:
        """
        Saves the current agent instance to a pickle file and update
        its priority.
        """
        agent = agent.to("cpu")
        joblib.dump(
            agent,
            filename=(
                f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_"
                f"{self.current_agent_index % self.max_agents}.pkl"
            ),
        )
        agent = agent.to(self.device)
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

    def get_agent(self, agent_id: int) -> Agent:
        """
        Samples an agent id from the softmax distribution induced by winrates
        then loads the selected agent from the buffer of saved agents.
        """
        if self.loaded_agents[agent_id] is None:
            if self.n_loaded_agents >= self.max_loaded_agents:
                pass
            print(
                "get agent",
                agent_id,
                f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_{agent_id}.pkl",
            )
            self.loaded_agents[agent_id] = joblib.load(
                f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_{agent_id}.pkl"
            ).to(self.device)

        return self.loaded_agents[agent_id]

    def _sample_distribution(self) -> List[float]:
        logits = [
            self.history[agent_id]["winrate"]
            for agent_id in range(self.current_agent_index)
        ]
        return softmax(logits)

    def sample(self, k=1):
        """ Samples k agents from the buffer of saved agents, prioritizing opponents with higher winrates."""
        probs = self._sample_distribution()
        return list(np.random.choice(
            np.arange(self.current_agent_index),
            p=probs,
            size=k,
        ))

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """Log metrics for the curriculum."""
        logs.append("winrate", self.history["winrate"])
        logs.append("games_played", self.history["n_games"])
        logs.append("stored_agents", self.n_loaded_agents)
        super().log_metrics(writer, logs, step, log_n_tasks)
