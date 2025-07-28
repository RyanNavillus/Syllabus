import copy
import warnings
from collections import defaultdict
from io import BytesIO
from multiprocessing.shared_memory import ShareableList
from threading import Lock
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch import Tensor

from syllabus.core import Curriculum
from syllabus.task_space.task_space import TaskSpace
from syllabus.utils import UsageError, compress_ranges

Array = Union[np.ndarray, Tensor]
RecurrentState = Union[Tuple[Array, Array], Array]


class Evaluator:
    """An interface for evaluating a trained agent, used by several curricula."""

    def __init__(
        self,
        agent: Any,
        device: torch.device = "cpu",
        preprocess_obs: Optional[Callable] = None,
        copy_agent: bool = True,
        simple_copy: bool = False,
        task_space: Optional[TaskSpace] = None,
        eval_envs: Optional[VectorEnv] = None,
        eval_curriculum: Optional[Curriculum] = None,
        recurrent_method: Optional[str] = None,
        recurrent_size: Optional[int] = None,
    ):
        """
        Initialize the Evaluator.

        Args:
            agent (Any): The trained agent to be evaluated.
            device (Optional[torch.device]): The device to run the evaluation on.
            preprocess_obs (Optional[Any]): A function to preprocess observations.
            copy_agent (bool): Whether to make a copy of the agent.
        """
        self._agent_reference = agent
        self.device = device
        self.preprocess_obs = preprocess_obs
        self._copy_agent = copy_agent   # Save to skip update if possible
        assert not (simple_copy and not copy_agent), "Cannot use simple_copy without copy_agent being True."

        # Make cpu copy of model
        if copy_agent and not simple_copy:
            try:
                # Save agent in memory
                model_data_in_memory = BytesIO()
                torch.save(self._agent_reference, model_data_in_memory, pickle_protocol=-1)
                model_data_in_memory.seek(0)

                # Load the model from memory to CPU
                self.agent = torch.load(model_data_in_memory, map_location=self.device, weights_only=False)
                model_data_in_memory.close()
            except RuntimeError as e:
                warnings.warn(str(e), stacklevel=2)
                simple_copy = True

        if copy_agent and simple_copy:
            agent.to(self.device)
            self.agent = copy.deepcopy(agent).to(self.device)
            agent.to("cuda")

        if not simple_copy:
            self.agent = self._agent_reference

        self.task_space = task_space if task_space is not None else eval_curriculum.task_space if eval_curriculum is not None else None
        self.eval_envs = eval_envs
        self.eval_curriculum = eval_curriculum
        self.recurrent_method = recurrent_method
        self.recurrent_size = recurrent_size

        try:
            import pettingzoo
            self.is_multiagent = isinstance(self.eval_envs, (
                pettingzoo.utils.BaseWrapper,
                pettingzoo.utils.BaseParallelWrapper,
                pettingzoo.AECEnv,
                pettingzoo.ParallelEnv
            ))
        except ImportError:
            self.is_multiagent = False

    def _update_agent(self):
        """
        Update the agent with a copy of the agent reference.
        This is necessary if you are using a model with different training and evaluation modes
        because the evaluator may need to run in eval mode while the agent is training.
        """
        if self._copy_agent:
            # Copy most recent parameters from agent reference
            self.agent.load_state_dict(self._agent_reference.state_dict())

    def get_action(
        self, state: Array, recurrent_state: RecurrentState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Sample an action from the policy for a given environment state.

        Args:
            state (Array): The current environment state.
            recurrent_state (Optional[RecurrentState]): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The action and additional information.
        """
        self._update_agent()
        state = self._prepare_state(state)
        if recurrent_state is not None:
            recurrent_state, done = self._prepare_recurrent(recurrent_state, done)

        self._set_eval_mode()
        with torch.no_grad():
            action, recurrent_state, extras = self._get_action(
                state, recurrent_state=recurrent_state, done=done
            )
        self._set_train_mode()
        return action, recurrent_state, extras

    def get_value(
        self, state: Array, recurrent_state: RecurrentState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get the value of a given environment state.

        Args:
            state (Array): The current environment state.
            recurrent_state (Optional[RecurrentState] ): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The value and additional information.
        """
        self._update_agent()
        state = self._prepare_state(state)
        if recurrent_state is not None:
            recurrent_state, done = self._prepare_recurrent(recurrent_state, done)
        self._set_eval_mode()
        with torch.no_grad():
            value, recurrent_state, extras = self._get_value(
                state, recurrent_state=recurrent_state, done=done
            )
        self._set_train_mode()
        return value, recurrent_state, extras

    def get_action_and_value(
        self, state: Array, recurrent_state: RecurrentState = None, done: Optional[Array] = None
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Get the action and value for a given environment state.

        Args:
            state (Array): The current environment state.
            recurrent_state (Optional[RecurrentState]): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: The action, value, and additional information.
        """
        self._update_agent()
        state = self._prepare_state(state)
        if recurrent_state is not None:
            recurrent_state, done = self._prepare_recurrent(recurrent_state, done)

        self._set_eval_mode()
        with torch.no_grad():
            action, value, recurrent_state, extras = self._get_action_and_value(
                state, recurrent_state=recurrent_state, done=done
            )
        self._set_train_mode()
        return action, value, recurrent_state, extras

    def _get_action(
        self, state: Array, recurrent_state: RecurrentState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get the action for a given environment state.

        Args:
            state (Array): The current environment state.
            recurrent_state (Optional[RecurrentState]): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, RecurrentState, Dict[str, Any]]: The action and additional information.
        """
        raise NotImplementedError

    def _get_value(
        self, state: Array, recurrent_state: RecurrentState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Abstract method to get the value of a given environment state.
        Can be overridden to interface with different agent implementations.

        Args:
            state (Array): The current environment state.
            recurrent_state (Optional[RecurrentState]): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, RecurrentState, Dict[str, Any]]: The value and additional information.
        """
        raise NotImplementedError

    def _get_action_and_value(
        self, state: Array, recurrent_state: RecurrentState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Abstract method to get the action and value for a given state.

        Args:
            state (Array): The current state.
            recurrent_state (Optional[recurrent_state]): The recurrent state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, RecurrentState, Dict[str, Any]]: The action, value, and additional information.
        """
        action, _, _ = self._get_action(state, recurrent_state, done)
        value, recurrent_state, extras = self._get_value(state, recurrent_state, done)
        return action, value, recurrent_state, extras

    def _prepare_state(self, state: Array) -> torch.Tensor:
        """
        Prepare the state for evaluation.

        Args:
            state (Array): The current state.

        Returns:
            torch.Tensor: The prepared state.
        """
        state = torch.Tensor(np.stack(state))
        if self.preprocess_obs is not None:
            state = self.preprocess_obs(state)
        state = state.to(self.device)
        return state

    def _prepare_recurrent(
        self, recurrent_state: RecurrentState, done: Array
    ) -> Tuple[RecurrentState, torch.Tensor]:
        """
        Prepare the recurrent state and done flag for evaluation.

        Args:
            recurrent_state (Tuple[Any, Any]): The recurrent state.
            done (Any): The done flag.

        Returns:
            Tuple[RecurrentState, torch.Tensor]: The prepared recurrent state and done flag.
        """
        if self.recurrent_method == "lstm":
            if self.recurrent_method == "lstm":
                assert (
                    isinstance(recurrent_state, tuple)
                    and len(recurrent_state) == 2
                    and isinstance(recurrent_state[0], Array)
                    and isinstance(recurrent_state[1], Array)
                ), (
                    "Recurrent state must be a tuple of "
                    "(cell_state, hidden_state) for LSTM."
                )
                recurrent_state = (
                    torch.Tensor(recurrent_state[0]).to(self.device),
                    torch.Tensor(recurrent_state[1]).to(self.device),
                )
            recurrent_state = (
                torch.Tensor(recurrent_state[0]).to(self.device),
                torch.Tensor(recurrent_state[1]).to(self.device),
            )
        elif self.recurrent_method == "rnn":
            assert isinstance(recurrent_state, Array), "Recurrent state must be a tensor for RNN."
            recurrent_state = recurrent_state.to(self.device)

        done = torch.Tensor(done)
        done = done.to(self.device)
        return recurrent_state, done

    def _set_eval_mode(self):
        """
        Set the policy to evaluation mode.
        """

    def _set_train_mode(self):
        """
        Set the policy to training mode.
        """

    def _initial_recurrent_state(self, batch_size):
        if self.recurrent_method == "lstm":
            return (
                torch.zeros(1, batch_size, self.recurrent_size),
                torch.zeros(1, batch_size, self.recurrent_size)
            )
        elif self.recurrent_method == "rnn":
            return torch.zeros(batch_size, self.recurrent_size)
        else:
            return None

    def evaluate_batch(self, steps, initial_obs, recurrent_state=None, rewards=None, dones=None, tasks=None, value_preds=None):
        """
        Evaluate the agent over a batch of steps.

        Args:
            steps (List[Tuple[Array, Array]]): List of (observation, recurrent_state) pairs.
            initial_obs (Array): Initial observations for the batch.
            recurrent_state (Optional[RecurrentState]): Current recurrent state.
            rewards (Optional[Array]): Array to store rewards.
            dones (Optional[Array]): Array to store done flags.
            tasks (Optional[Array]): Array to store tasks.
            value_preds (Optional[Array]): Array to store value predictions.

        Returns:
            Tuple[Array, RecurrentState, Array, Array, Array, Array]: Updated observations, recurrent state, rewards, dones, tasks, and value predictions.
        """
        assert not self.is_multiagent, "Batch evaluation is not supported for multi-agent environments."
        num_envs = self.eval_envs.num_envs

        # Standard RL data
        obs = initial_obs
        recurrent_state = recurrent_state if recurrent_state is not None else self._initial_recurrent_state(num_envs)
        rewards = rewards if rewards is not None else torch.zeros((steps, num_envs))
        dones = dones if dones is not None else torch.zeros((steps, num_envs))
        tasks = tasks if tasks is not None else torch.zeros((steps, num_envs))
        value_preds = value_preds if value_preds is not None else torch.zeros((steps, num_envs))

        for i in range(steps):
            actions, value, recurrent_state, _ = self.get_action_and_value(obs, recurrent_state, done=dones)
            if isinstance(self.eval_envs.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
                actions = actions.int()
            obs, rew, term, trunc, info = self.eval_envs.step(actions.cpu().numpy())

            done = torch.logical_or(torch.Tensor(term), torch.Tensor(trunc))
            rewards[i] = torch.Tensor(rew)
            dones[i + 1] = torch.Tensor(done)
            tasks[i] = torch.Tensor(info["task"])
            value_preds[i] = torch.squeeze(value.cpu())

        return obs, recurrent_state, rewards, dones, tasks, value_preds

    def evaluate_agent(self, num_episodes=100, verbose=False, store_all=False):
        """
        Evaluate the agent over a number of episodes.

        Args:
            num_episodes (int): Number of episodes to evaluate.
            verbose (bool): Whether to print episode results.
            store_all (bool): Whether to store all step data.

        Returns:
            Tuple[List[float], torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]: Returns, task success rates, final task success rates, and optional step data.
        """
        assert self.task_space is not None, "Task space must be defined for evaluation."
        if self.is_multiagent:
            return self._evaluate_pettingzoo(num_episodes, verbose=verbose, store_all=store_all)
        else:
            return self._evaluate_gymnasium(num_episodes, verbose=verbose, store_all=store_all)

    def _evaluate_gymnasium(self, num_episodes=100, verbose=False, store_all=False):
        """
        Evaluate the agent over a number of episodes.

        Args:
            num_episodes (int): Number of episodes to evaluate.

        Returns:
            List[float]: List of returns for each episode.
        """
        # Testing
        num_envs = self.eval_envs.num_envs

        # Standard RL data
        obs, info = self.eval_envs.reset()
        recurrent_state = self._initial_recurrent_state(num_envs)
        completed_episodes = 0
        returns = []
        rews = torch.zeros(num_envs)
        dones = [False] * num_envs

        if store_all:
            step_rewards = []
            step_value_preds = []
            step_dones = []
            step_tasks = []

        # Track task progress
        task_counts = torch.zeros(self.task_space.num_tasks, dtype=int)
        task_successes = torch.zeros(self.task_space.num_tasks, dtype=float)

        final_task_counts = torch.zeros(self.task_space.num_tasks, dtype=int)
        final_task_successes = torch.zeros(self.task_space.num_tasks, dtype=float)

        while completed_episodes < num_episodes:
            if store_all:
                actions, value, recurrent_state, _ = self.get_action_and_value(obs, recurrent_state, done=dones)
            else:
                actions, recurrent_state, _ = self.get_action(obs, recurrent_state, done=dones)
            if isinstance(self.eval_envs.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
                actions = actions.int()
            obs, rew, term, trunc, info = self.eval_envs.step(actions.cpu().numpy())
            rews += rew
            dones = np.logical_or(term, trunc)

            if store_all:
                step_rewards.append(torch.Tensor(rew).unsqueeze(-1))
                step_value_preds.append(value.cpu())
                step_dones.append(torch.zeros_like(value.cpu()))
                step_tasks.append(torch.Tensor(info["task"]))

            # Track task completion for tasks that can change mid-episode
            if "task_completion" in info:
                task_idx = info["task"]
                task_completions = info["task_completion"]

                for task, completion in zip(task_idx, task_completions):
                    # Assumes that success will only remain 1.0 or -1.0 for 1 step
                    if completion >= 1.0:
                        # Task has succeeded
                        task_counts[task] += 1
                        task_successes[task] += 1.0
                    elif completion < 0:
                        # Task has failed
                        task_counts[task] += 1
                        task_successes[task] += 0.0

            # Handle episode completion
            for i, done in enumerate(dones):
                if done:
                    if verbose:
                        print(f"Episode {completed_episodes} finished for task {info['task'][i]} with return {rews[i]}")
                    returns.append(rews[i])
                    rews[i] = 0
                    completed_episodes += 1

                    # Track task progress at end of episode
                    completion = info["task_completion"][i]
                    task = info["task"][i]
                    final_task_counts[task] += 1
                    final_task_successes[task] += max(completion, 0.0)

        if torch.any(final_task_counts == 0):
            warnings.warn(
                f"Tasks {compress_ranges(torch.where(task_counts == 0)[0].tolist())} were not attempted during evaluation. Consider increasing eval episodes.")
        task_counts = torch.maximum(task_counts, torch.ones_like(task_counts))
        final_task_counts = torch.maximum(final_task_counts, torch.ones_like(final_task_counts))
        task_success_rates = torch.divide(task_successes, task_counts)
        final_task_success_rates = torch.divide(final_task_successes, final_task_counts)
        all_data = None
        if store_all:
            all_data = {
                "rewards": step_rewards,
                "value_preds": step_value_preds,
                "dones": step_dones,
                "tasks": step_tasks
            }
        return returns, task_success_rates, final_task_success_rates, all_data

    def _evaluate_pettingzoo(self, num_episodes=100, verbose=False):
        """
        Evaluate the agent over a number of episodes.

        Args:
            num_episodes (int): Number of episodes to evaluate.

        Returns:
            List[float]: List of returns for each episode.
        """
        # Testing
        num_envs = self.eval_envs.num_envs

        # Standard RL data
        obs, info = self.eval_envs.reset()
        recurrent_state = self._initial_recurrent_state(num_envs)
        completed_episodes = 0
        returns = []
        rews = torch.zeros(num_envs)
        dones = [False] * num_envs

        # Track task progress
        task_counts = torch.zeros(self.task_space.num_tasks, dtype=int)
        task_successes = torch.zeros(self.task_space.num_tasks, dtype=float)

        final_task_counts = torch.zeros(self.task_space.num_tasks, dtype=int)
        final_task_successes = torch.zeros(self.task_space.num_tasks, dtype=float)

        while completed_episodes < num_episodes:
            actions, recurrent_state, _ = self.get_action(obs, recurrent_state, done=dones)
            if isinstance(self.eval_envs.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
                actions = actions.int()
            obs, rew, terms, truncs, infos = self.eval_envs.step(actions.cpu().numpy())
            rews += rew
            dones = tuple(
                {k: a or b for k, a, b in zip(term.keys(), term.values(), trunc.values())}
                for term, trunc in zip(terms, truncs)
            )
            all_dones = [all(list(done.values())) for done in dones]

            # Track task completion for tasks that can change mid-episode
            if isinstance(infos, list) and "task_completion" in infos[0]:
                task_idx = [i["task"] for i in infos]
                task_completions = [i["task_completion"] for i in infos]

                for task, completion in zip(task_idx, task_completions):
                    # Assumes that success will only remain 1.0 or -1.0 for 1 step
                    if completion >= 1.0:
                        # Task has succeeded
                        task_counts[task] += 1
                        task_successes[task] += 1.0
                    elif completion < 0:
                        # Task has failed
                        task_counts[task] += 1
                        task_successes[task] += 0.0
            elif isinstance(infos, list) and "task_completion" in infos[0]:
                task_completions = [info["task_completion"] for info in infos]
                task_idx = [info["task"] for info in infos]
            else:
                raise UsageError(
                    "Did not find 'task_completion' in infos. Task success rates will not be evaluated.")

            # Handle episode completion
            for i, done in enumerate(all_dones):
                if done:
                    if verbose:
                        print(f"Episode {completed_episodes} finished for task {info['task'][i]} with return {rews[i]}")
                    returns.append(rews[i])
                    rews[i] = 0
                    completed_episodes += 1

                    # Track task progress at end of episode
                    task_completions = [i["task_completion"] for i in infos]
                    task_idx = [i["task"] for i in infos]
                    task = task_idx[i]
                    final_task_counts[task] += 1
                    final_task_successes[task] += max(task_completions[i], 0.0)

        if torch.any(final_task_counts == 0):
            warnings.warn(
                f"Tasks {compress_ranges(torch.where(task_counts == 0)[0].tolist())} were not attempted during evaluation. Consider increasing eval episodes.")
        task_counts = torch.maximum(task_counts, torch.ones_like(task_counts))
        final_task_counts = torch.maximum(final_task_counts, torch.ones_like(final_task_counts))
        task_success_rates = torch.divide(task_successes, task_counts)
        final_task_success_rates = torch.divide(final_task_successes, final_task_counts)
        return returns, task_success_rates, final_task_success_rates, None


class DummyEvaluator(Evaluator):
    def __init__(self, action_space, *args, **kwargs):
        self.action_space = action_space
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_shape = 1
        else:
            self.action_shape = action_space.sample().shape
        kwargs.pop("copy_agent", None)
        super().__init__(None, *args, copy_agent=False, **kwargs)

    def _get_state_shape(self, state):
        if isinstance(state, (torch.Tensor, np.ndarray)):
            state_shape = state.shape[0]
        elif isinstance(state, (list, tuple)):
            state_shape = len(state)
        else:
            state_shape = 1
        return state_shape

    def _get_action(self, state, recurrent_state=None, done=None):
        state_shape = self._get_state_shape(state)
        recurrent_state = self._initial_recurrent_state() if recurrent_state is not None else None
        return torch.zeros((state_shape, self.action_shape)), recurrent_state, {}

    def _get_value(self, state, recurrent_state=None, done=None):
        state_shape = self._get_state_shape(state)
        recurrent_state = self._initial_recurrent_state() if recurrent_state is not None else None
        return torch.zeros((state_shape, 1)), recurrent_state, {}

    def _get_action_and_value(self, state, recurrent_state=None, done=None):
        state_shape = self._get_state_shape(state)
        recurrent_state = self._initial_recurrent_state() if recurrent_state is not None else None
        return torch.zeros((state_shape, 1)), torch.zeros((state_shape, self.action_shape)), recurrent_state, {}

    def evaluate_agent(self, num_episodes=100, verbose=False):
        return np.zeros(num_episodes), np.zeros(self.task_space.num_tasks), np.zeros(self.task_space.num_tasks)


class CleanRLEvaluator(Evaluator):
    def _get_action(self, state, recurrent_state=None, done=None):
        if self.recurrent_method is not None:
            self._check_inputs(recurrent_state, done)
            action, recurrent_state = self.agent.get_action(state, recurrent_state, done)
        else:
            action = self.agent.get_action(state)
            recurrent_state = None
        return action, recurrent_state, {}

    def _get_value(self, state, recurrent_state=None, done=None):
        if self.recurrent_method is not None:
            self._check_inputs(recurrent_state, done)
            value, recurrent_state = self.agent.get_value(state, recurrent_state, done)
        else:
            value = self.agent.get_value(state)
            recurrent_state = None
        return value, recurrent_state, {}

    def _get_action_and_value(self, state, recurrent_state=None, done=None):
        if self.recurrent_method is not None:
            self._check_inputs(recurrent_state, done)
            action, log_probs, entropy, value, recurrent_state = self.agent.get_action_and_value(
                state, recurrent_state, done)
        else:
            action, log_probs, entropy, value = self.agent.get_action_and_value(state)
            recurrent_state = None
        return action, value, recurrent_state, {"log_probs": log_probs, "entropy": entropy}

    def _check_inputs(self, recurrent_state, done):
        assert (
            recurrent_state is not None
        ), "Recurrent state must be provided. Make sure to configure any recurrence-specific settings for your curriculum."
        assert (
            done is not None
        ), "Done must be provided. Make sure to configure any recurrence-specific settings for your curriculum."
        return True

    def _set_eval_mode(self):
        self.agent.eval()

    def _set_train_mode(self):
        self.agent.train()


class MoolibEvaluator(Evaluator):
    def __init__(self, agent, *args, **kwargs):
        super().__init__(agent, *args, **kwargs)

    def _get_action(self, state, recurrent_state=None, done=None):
        self._check_inputs(recurrent_state, done)
        state["done"] = done
        output, recurrent_state = self.agent(state, recurrent_state, get_action=True, get_value=False)
        action = output["action"]
        return action, recurrent_state, {}

    def _get_value(self, state, recurrent_state=None, done=None):
        self._check_inputs(recurrent_state, done)
        state["done"] = done
        output, recurrent_state = self.agent(state, recurrent_state, get_action=False, get_value=True)
        value = output["baseline"].reshape(-1, 1)
        return value, recurrent_state, {}

    def _get_action_and_value(self, state, recurrent_state=None, done=None):
        self._check_inputs(recurrent_state, done)
        state["done"] = done
        output, recurrent_state = self.agent(state, recurrent_state, get_action=True, get_value=True)
        action = output["action"]
        value = output["baseline"].reshape(-1, 1)
        return action, value, recurrent_state, {}

    def _prepare_state(self, state) -> torch.Tensor:
        full_dict = defaultdict(list)
        if isinstance(state, list):
            for obs_dict in state:
                for k, v in obs_dict.items():
                    full_dict[k].append(v)
        elif isinstance(state, dict):
            full_dict = state
        tensor_dict = {key: torch.unsqueeze(torch.Tensor(np.stack(val_list)), 0).to(self.device)
                       for key, val_list in full_dict.items()}
        return tensor_dict

    def _check_inputs(self, recurrent_state, done):
        assert (
            recurrent_state is not None
        ), "Recurrent state must be provided. Make sure to configure any recurrence-specific settings for your curriculum."
        return True

    def _set_eval_mode(self):
        self.agent.eval()

    def _set_train_mode(self):
        self.agent.train()


class GymnasiumEvaluationWrapper(gym.Wrapper):
    instance_lock = Lock()
    env_count = ShareableList([0])

    def __init__(
        self,
        *args,
        task_space: TaskSpace = None,
        change_task_on_completion: bool = False,
        eval_only_n_tasks: bool = None,
        ignore_seed: bool = False,
        randomize_order: bool = True,
        start_index_spacing: int = 0,
        **kwargs
    ):
        if start_index_spacing > 0:
            with GymnasiumEvaluationWrapper.instance_lock:
                instance_id = GymnasiumEvaluationWrapper.env_count[0]
                GymnasiumEvaluationWrapper.env_count[0] += 1

        super().__init__(*args, **kwargs)
        self.change_task_on_completion = change_task_on_completion
        self.task_space = task_space if task_space is not None else self.env.task_space
        self.tidx = (start_index_spacing * instance_id) % len(self.task_space.tasks) if start_index_spacing > 0 else 0
        eval_only_n_tasks = eval_only_n_tasks if eval_only_n_tasks is not None else self.task_space.num_tasks
        self.random_tasks = copy.deepcopy(self.task_space.tasks[:eval_only_n_tasks])
        if randomize_order:
            if ignore_seed:
                rng = np.random.default_rng()
                rng.shuffle(self.random_tasks)
            else:
                np.random.shuffle(self.random_tasks)

    def reset(self, **kwargs):
        new_task = self.random_tasks[self.tidx]

        # Repeat task list when done
        self.tidx = (self.tidx + 1) % len(self.random_tasks)
        obs, info = self.env.reset(new_task=new_task, **kwargs)
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if "task_completion" in info and (info["task_completion"] >= 1.0 or info["task_completion"] < 0) and self.change_task_on_completion and not (term or trunc):
            new_task = self.random_tasks[self.tidx]
            self.tidx = (self.tidx + 1) % len(self.random_tasks)
            self.env.change_task(new_task)
        return obs, rew, term, trunc, info
