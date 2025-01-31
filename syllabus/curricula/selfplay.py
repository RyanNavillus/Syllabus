""" Self play curricula for training agents against copies themselves. This is an experimental API and subject to change."""

import os
import time
from copy import deepcopy
from typing import List
from collections import OrderedDict
import joblib
import numpy as np
from gymnasium import spaces
from scipy.special import softmax
from queue import Queue

from syllabus.core import Agent, Curriculum  # noqa: E402
from syllabus.task_space import DiscreteTaskSpace  # noqa: E402


class WinrateBuffer:
    """
    Stores the winrate of each agent in a queue and provides a sampling distribution.
    """

    def __init__(
        self,
        max_agents: int,
        entropy_parameter: float,
        smoothing_constant: int,
        buffer_size: int = 128,
    ):
        self.max_agents = max_agents
        self.buffer_size = buffer_size
        self.buffer = {i: Queue(maxsize=buffer_size) for i in range(max_agents)}
        self.entropy_parameter = entropy_parameter
        self.smoothing_constant = smoothing_constant
        self.initialized_agents = np.zeros(max_agents)

    def update_winrate(self, agent_id: int, reward: float):
        print(reward)
        print(agent_id)
        reward = reward == 1  # converts rewards {-1;1} to winrate {0;1}
        self.buffer[agent_id].put(reward)
        if self.buffer[agent_id].full():
            self.buffer[agent_id].get()

        # mark agent as initialized
        # unitiliazed agents will be masked from the sampling distribution
        if not self.initialized_agents[agent_id]:
            self.initialized_agents[agent_id] = 1

    def get_winrate(self, agent_id: int):
        # TODO: should we return a winrate if the queue is not full?
        if self.buffer[agent_id].empty():
            return 0.0
        return np.mean(self.buffer[agent_id].queue)

    def _apply_entropy(self, winrate: float):
        if np.isnan(winrate):
            return 0.0
        return winrate**self.entropy_parameter

    def get_sampling_distribution(self):
        """
        Return a sampling distribution reflecting the difficulty of each opponent.
        Uninitialized agents are masked and not included in the distribution.
        """
        loss_rates = np.array([1 - self.get_winrate(i) for i in range(self.max_agents)])

        # mask uninitialized agents
        masked_loss_rates = np.ma.masked_array(
            loss_rates, mask=self.initialized_agents == 0
        )

        # apply the entropy function, smoothing and normalization to all valid loss rates
        masked_loss_rates = np.ma.array(
            [self._apply_entropy(winrate) for winrate in masked_loss_rates]
        )
        masked_loss_rates += self.smoothing_constant
        masked_sampling_distribution = masked_loss_rates / masked_loss_rates.sum()

        # unmask and set masked values to 0
        sampling_distribution = np.where(
            masked_sampling_distribution.mask, 0, masked_sampling_distribution
        )

        # if no agents are initialized, sample the first agent
        # this happens when the first agent has not yet receiveda reward
        if sampling_distribution.sum() == 0:
            sampling_distribution = np.zeros(self.max_agents)
            sampling_distribution[0] = 1.0

        return sampling_distribution

    def __repr__(self):
        return {i: self.get_winrate(i) for i in range(self.max_agents)}.__repr__()

    def __getitem__(self, agent_id):
        return self.get_winrate(agent_id)


class FIFOAgentBuffer:
    """
    First-In-First-Out buffer implemented as an OrderedDict.
    """

    def __init__(
        self,
        max_agents: int,
        curriculum_name: str,
        device: str,
        storage_path: str,
        seed: int,
    ):
        self.max_agents = max_agents
        self.curriculum_name = curriculum_name
        self.device = device
        self.storage_path = storage_path
        self.seed = seed
        self.buffer = OrderedDict()

    def add_agent(self, agent_id: int, agent: Agent) -> None:
        # Remove first item so that buffer length does not exceed max_agents
        if len(self.buffer) >= self.max_agents:
            self.buffer.popitem(last=False)
        self.buffer[agent_id] = agent
        # Move recently accessed agent to end of buffer (last to be removed)
        self.buffer.move_to_end(agent_id)

    def get_agent(self, agent_id: int) -> Agent:
        if agent_id not in self.buffer:
            # Delete first so that buffer length does not exceed max_agents
            if len(self.buffer) >= self.max_agents:
                self.buffer.popitem(last=False)
            print(
                "load agent",
                agent_id,
                f"{self.storage_path}/{self.curriculum_name}_{self.seed}_agent_checkpoint_{agent_id}.pkl",
            )
            self.buffer[agent_id] = joblib.load(
                f"{self.storage_path}/{self.curriculum_name}_{self.seed}_agent_checkpoint_{agent_id}.pkl"
            ).to(self.device)

        return self.buffer[agent_id]

    def __getitem__(self, agent_id):
        return self.buffer.get(agent_id, None)

    def __contains__(self, key):
        return key in self.buffer

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return self.buffer.__repr__()


class SelfPlay(Curriculum):
    """Self play curriculum for training agents against themselves."""

    def __init__(
        self,
        task_space: DiscreteTaskSpace,
        agent: Agent,
        device: str,
    ):
        """Initialize the self play curriculum.

        :param task_space: The task space of the environment
        :param agent: The initial agent to play against
        :param device: The device to run the agent on
        """
        # Self play can only return agent_id == 0
        assert isinstance(
            task_space, DiscreteTaskSpace) and task_space.num_tasks == 1, "Self play only supports DiscreteTaskSpaces with a single element."
        super().__init__(task_space)
        self.device = device
        self.agent = deepcopy(agent).to(self.device)
        self.task_space = DiscreteTaskSpace(1)  # SelfPlay can only return agent_id = 0
        self.history = {
            "winrate": 0,
            "n_games": 0,
        }

    def add_agent(self, agent: Agent) -> int:
        # TODO: Perform copy in RAM instead of VRAM
        self.agent = deepcopy(agent).to(self.device)
        return 0

    def get_agent(self, agent_id: int) -> Agent:
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
        # TODO: Is this formula correct?
        # I think it should be ((old_winrate * n) + win) / (n+1) (where n is the value before you add 1)
        # old_winrate * n is the old # of wins. Then add win to get the new number of wins. Divide by the new number of games.
        self.history["winrate"] = old_winrate + (win - old_winrate) / n

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """Log metrics for the curriculum."""
        logs.append("winrate", self.history["winrate"])
        logs.append("n_games", self.history["n_games"])
        super().log_metrics(writer, logs, step, log_n_tasks)


class FictitiousSelfPlay(Curriculum):

    def __init__(
        self,
        task_space: DiscreteTaskSpace,
        agent: Agent,
        device: str,
        storage_path: str,
        max_agents: int,
        seed: int = 0,
        max_loaded_agents: int = 10,
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
        self.task_space = DiscreteTaskSpace(self.max_agents)
        self.loaded_agents = FIFOAgentBuffer(
            max_loaded_agents, self.__class__.__name__, device, storage_path, seed
        )
        self.history = {
            "winrate": 0,
            "n_games": 0,
        }
        self.max_loaded_agents = max_loaded_agents
        self.add_agent(agent)  # creates the initial opponent

    def add_agent(self, agent):
        """
        Saves the current agent instance to a pickle file and adds it to the loaded agents buffer.
        When the `max_agents` limit is met, older agent checkpoints are overwritten.
        """
        # TODO: Check that this doesn't move original agent to cpu
        agent = agent.to("cpu")
        joblib.dump(
            agent,
            filename=(
                f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_"
                f"{self.current_agent_index}.pkl"
            ),
        )
        agent = agent.to(self.device)
        self.loaded_agents.add_agent(self.current_agent_index, agent)
        self.current_agent_index += 1

    def get_agent(self, agent_id: int) -> Agent:
        """Loads an agent from the buffer of saved agents."""
        return self.loaded_agents.get_agent(agent_id)

    def _sample_distribution(self) -> List[float]:
        # Number of saved agents up to max_agents
        n_agents = min(self.current_agent_index, self.max_agents)
        return [
            1.0 / n_agents for _ in range(n_agents)
        ] + [0.0 for _ in range(self.max_agents - n_agents)]

    def sample(self, k=1):
        probs = self._sample_distribution()
        # max_agents below the highest agent index, but not below 0
        min_agent_id = max(0, self.current_agent_index - self.max_agents)
        sample = list(
            np.random.choice(
                np.arange(min_agent_id, min_agent_id + self.max_agents),
                p=probs,
                size=k,
            )
        )
        return sample

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
        # TODO: Is this formula correct?
        # I think it should be ((old_winrate * n) + win) / (n+1) (where n is the value before you add 1)
        # old_winrate * n is the old # of wins. Then add win to get the new number of wins. Divide by the new number of games.
        self.history["winrate"] = old_winrate + (win - old_winrate) / n

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """Log metrics for the curriculum."""
        logs.append("winrate", self.history["winrate"])
        logs.append("games_played", self.history["n_games"])
        logs.append("stored_agents", len(self.loaded_agents))
        super().log_metrics(writer, logs, step, log_n_tasks)


class PrioritizedFictitiousSelfPlay(Curriculum):

    def __init__(
        self,
        task_space: DiscreteTaskSpace,
        agent: Agent,
        device: str,
        storage_path: str,
        max_agents: int,
        entropy_parameter: float,
        smoothing_constant: int,
        seed: int = 0,
        max_loaded_agents: int = 10,
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
        self.task_space = DiscreteTaskSpace(self.max_agents)
        self.winrate_buffer = WinrateBuffer(
            max_agents, entropy_parameter, smoothing_constant
        )
        self.loaded_agents = FIFOAgentBuffer(
            max_loaded_agents, self.__class__.__name__, device, storage_path, seed
        )
        self.history = {
            "winrate": 0,
            "n_games": 0,
        }
        self.max_loaded_agents = max_loaded_agents
        self.add_agent(agent)  # creates the initial opponent

    def add_agent(self, agent):
        """
        Saves the current agent instance to a pickle file and adds it to the loaded agents buffer.
        When the `max_agents` limit is met, older agent checkpoints are overwritten.
        """
        # TODO: Check that this doesn't move original agent to cpu
        agent = agent.to("cpu")
        joblib.dump(
            agent,
            filename=(
                f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_"
                f"{self.current_agent_index}.pkl"
            ),
        )
        agent = agent.to(self.device)
        self.loaded_agents.add_agent(self.current_agent_index, agent)
        self.current_agent_index += 1

    def get_agent(self, agent_id: int) -> Agent:
        """
        Samples an agent id from the softmax distribution induced by winrates
        then loads the selected agent from the buffer of saved agents.
        """
        # TODO: add sampling from the distribution
        if self.loaded_agents[agent_id] is None:
            if len(self.loaded_agents) >= self.max_loaded_agents:
                pass
            print(
                "get agent",
                agent_id,
                f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_{agent_id}.pkl",
            )
            self.loaded_agents.add_agent(
                agent_id,
                joblib.load(
                    f"{self.storage_path}/{self.__class__.__name__}_{self.seed}_agent_checkpoint_{agent_id}.pkl"
                ).to(self.device)
            )

        return self.loaded_agents[agent_id]

    def sample(self, k=1):
        """
        Samples k agents from the buffer of saved agents, prioritizing opponents with higher winrates
        Samples values from [self.current_agent_index - self.max_agents, self.current_agent_index)
        """
        probs = self.winrate_buffer.get_sampling_distribution()
        # max_agents below the highest agent index, but not below 0
        min_agent_id = max(0, self.current_agent_index - self.max_agents)
        return np.random.choice(
            np.arange(min_agent_id, min_agent_id + self.max_agents),
            p=probs,
            size=k,
        ).tolist()

    def update_winrate(self, agent_id: int, reward: int) -> None:
        """
        Uses an incremental mean to update an agent's winrate. This assumes that reward
        is positive for a win and negative for a loss. Not used for sampling.

        :param agent_id: Identifier of the agent
        :param reward: Reward received by the agent
        """
        self.winrate_buffer.update_winrate(agent_id, reward)

        win = reward > 0  # converts the reward to 0 or 1
        self.history["n_games"] += 1
        old_winrate = self.history["winrate"]
        n = self.history["n_games"]
        # TODO: Is this formula correct?
        # I think it should be ((old_winrate * n) + win) / (n+1) (where n is the value before you add 1)
        # old_winrate * n is the old # of wins. Then add win to get the new number of wins. Divide by the new number of games.
        self.history["winrate"] = old_winrate + (win - old_winrate) / n

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """Log metrics for the curriculum."""
        logs.append("winrate", self.history["winrate"])
        logs.append("games_played", self.history["n_games"])
        logs.append("stored_agents", len(self.loaded_agents))
        super().log_metrics(writer, logs, step, log_n_tasks)
