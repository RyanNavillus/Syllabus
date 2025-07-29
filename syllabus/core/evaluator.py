import copy
from threading import Lock
from multiprocessing.shared_memory import ShareableList

import warnings
from collections import defaultdict
from io import BytesIO
from typing import Any, Callable, Dict, Optional, Tuple, Union
import warnings

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from syllabus.task_space.task_space import TaskSpace


Array = Union[np.ndarray, Tensor]
LSTMState = Tuple[Array, Array]


class Evaluator:
    """An interface for evaluating a trained agent, used by several curricula."""

    def __init__(
        self,
        agent: Any,
        device: Optional[torch.device] = None,
        preprocess_obs: Optional[Callable] = None,
        copy_agent: bool = True,
        simple_copy: bool = False,
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
            self.agent.require_grad_(False)
            agent.to("cuda")

        if not simple_copy:
            self.agent = self._agent_reference

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
        self, state: Array, lstm_state: LSTMState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Sample an action from the policy for a given environment state.

        Args:
            state (Array): The current environment state.
            lstm_state (Optional[LSTMState]): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The action and additional information.
        """
        self._update_agent()
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)

        self._set_eval_mode()
        with torch.no_grad():
            action, lstm_state, extras = self._get_action(
                state, lstm_state=lstm_state, done=done
            )
        self._set_train_mode()
        return action, lstm_state, extras

    def get_value(
        self, state: Array, lstm_state: LSTMState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get the value of a given environment state.

        Args:
            state (Array): The current environment state.
            lstm_state (Optional[LSTMState] ): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The value and additional information.
        """
        self._update_agent()
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)
        self._set_eval_mode()
        with torch.no_grad():
            value, lstm_state, extras = self._get_value(
                state, lstm_state=lstm_state, done=done
            )
        self._set_train_mode()
        return value, lstm_state, extras

    def get_action_and_value(
        self, state: Array, lstm_state: LSTMState = None, done: Optional[Array] = None
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """
        Get the action and value for a given environment state.

        Args:
            state (Array): The current environment state.
            lstm_state (Optional[LSTMState]): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: The action, value, and additional information.
        """
        self._update_agent()
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)

        self._set_eval_mode()
        with torch.no_grad():
            action, value, lstm_state, extras = self._get_action_and_value(
                state, lstm_state=lstm_state, done=done
            )
        self._set_train_mode()
        return action, value, lstm_state, extras

    def _get_action(
        self, state: Array, lstm_state: LSTMState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get the action for a given environment state.

        Args:
            state (Array): The current environment state.
            lstm_state (Optional[LSTMState]): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, LSTMState, Dict[str, Any]]: The action and additional information.
        """
        raise NotImplementedError

    def _get_value(
        self, state: Array, lstm_state: LSTMState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Abstract method to get the value of a given environment state.
        Can be overridden to interface with different agent implementations.

        Args:
            state (Array): The current environment state.
            lstm_state (Optional[LSTMState]): The LSTM cell and hidden state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, LSTMState, Dict[str, Any]]: The value and additional information.
        """
        raise NotImplementedError

    def _get_action_and_value(
        self, state: Array, lstm_state: LSTMState = None, done: Optional[Array] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Abstract method to get the action and value for a given state.

        Args:
            state (Array): The current state.
            lstm_state (Optional[LSTM_state]): The LSTM state.
            done (Optional[Array]): The done flag.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, LSTMState, Dict[str, Any]]: The action, value, and additional information.
        """
        action, _, _ = self._get_action(state, lstm_state, done)
        value, lstm_state, extras = self._get_value(state, lstm_state, done)
        return action, value, lstm_state, extras

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
        if self.device is not None:
            state = state.to(self.device)
        return state

    def _prepare_lstm(
        self, lstm_state: LSTMState, done: Array
    ) -> Tuple[LSTMState, torch.Tensor]:
        """
        Prepare the LSTM state and done flag for evaluation.

        Args:
            lstm_state (Tuple[Any, Any]): The LSTM state.
            done (Any): The done flag.

        Returns:
            Tuple[LSTMState, torch.Tensor]: The prepared LSTM state and done flag.
        """
        lstm_state = (
            torch.Tensor(lstm_state[0]),
            torch.Tensor(lstm_state[1]),
        )
        done = torch.Tensor(done)
        if self.device is not None:
            lstm_state = (
                lstm_state[0].to(self.device),
                lstm_state[1].to(self.device),
            )
            done = done.to(self.device)
        return lstm_state, done

    def _set_eval_mode(self):
        """
        Set the policy to evaluation mode.
        """

    def _set_train_mode(self):
        """
        Set the policy to training mode.
        """


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

    def _get_action(self, state, lstm_state=None, done=None):
        state_shape = self._get_state_shape(state)
        lstm_state = (torch.zeros_like(lstm_state[0]), torch.zeros_like(
            lstm_state[1])) if lstm_state is not None else None
        return torch.zeros((state_shape, self.action_shape)), lstm_state, {}

    def _get_value(self, state, lstm_state=None, done=None):
        state_shape = self._get_state_shape(state)
        lstm_state = (torch.zeros_like(lstm_state[0]), torch.zeros_like(
            lstm_state[1])) if lstm_state is not None else None
        return torch.zeros((state_shape, 1)), lstm_state, {}

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        state_shape = self._get_state_shape(state)
        lstm_state = (torch.zeros_like(lstm_state[0]), torch.zeros_like(
            lstm_state[1])) if lstm_state is not None else None
        return torch.zeros((state_shape, 1)), torch.zeros((state_shape, self.action_shape)), lstm_state, {}


class CleanRLEvaluator(Evaluator):
    def __init__(self, agent, *args, is_lstm=False, **kwargs):
        super().__init__(agent, *args, **kwargs)
        self.is_lstm = is_lstm or hasattr(agent, "lstm")

    def _get_action(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            self._check_inputs(lstm_state, done)
            action, lstm_state = self.agent.get_action(state, lstm_state, done)
        else:
            action = self.agent.get_action(state)
            lstm_state = None
        return action, lstm_state, {}

    def _get_value(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            self._check_inputs(lstm_state, done)
            value, lstm_state = self.agent.get_value(state, lstm_state, done)
        else:
            value = self.agent.get_value(state)
            lstm_state = None
        return value, lstm_state, {}

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            self._check_inputs(lstm_state, done)
            action, log_probs, entropy, value, lstm_state = self.agent.get_action_and_value(state, lstm_state, done)
        else:
            action, log_probs, entropy, value = self.agent.get_action_and_value(state)
            lstm_state = None
        return action, value, lstm_state, {"log_probs": log_probs, "entropy": entropy}

    def _check_inputs(self, lstm_state, done):
        assert (
            lstm_state is not None
        ), "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
        assert (
            done is not None
        ), "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
        return True

    def _set_eval_mode(self):
        self.agent.eval()

    def _set_train_mode(self):
        self.agent.train()


class MoolibEvaluator(Evaluator):
    def __init__(self, agent, *args, **kwargs):
        super().__init__(agent, *args, **kwargs)

    def _get_action(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        state["done"] = done
        output, lstm_state = self.agent(state, lstm_state, get_action=True, get_value=False)
        action = output["action"]
        return action, lstm_state, {}

    def _get_value(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        state["done"] = done
        output, lstm_state = self.agent(state, lstm_state, get_action=False, get_value=True)
        value = output["baseline"].reshape(-1, 1)
        return value, lstm_state, {}

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        state["done"] = done
        output, lstm_state = self.agent(state, lstm_state, get_action=True, get_value=True)
        action = output["action"]
        value = output["baseline"].reshape(-1, 1)
        return action, value, lstm_state, {}

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

    def _check_inputs(self, lstm_state, done):
        assert (
            lstm_state is not None
        ), "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
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
