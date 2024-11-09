import copy
from collections import defaultdict
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

Array = Union[np.ndarray, Tensor]
LSTMState = Tuple[Array, Array]


class Evaluator:
    """An interface for evaluating a trained agent, used by several curricula."""

    def __init__(
        self,
        agent: Any,
        device: Optional[torch.device] = None,
        preprocess_obs: Optional[Callable] = None,
    ):
        """
        Initialize the Evaluator.

        Args:
            agent (Any): The trained agent to be evaluated.
            device (Optional[torch.device]): The device to run the evaluation on.
            preprocess_obs (Optional[Any]): A function to preprocess observations.
        """
        self._agent_reference = agent
        self.agent = None
        self.device = device
        self.preprocess_obs = preprocess_obs

    def _update_agent(self):
        """
        Update the agent with a copy of the agent reference.
        This is necessary if you are using a model with different training and evaluation modes
        because the evaluator may need to run in eval mode while the agent is training.
        """
        # Do not make a copy by default
        self.agent = self._agent_reference

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
        return value.to("cpu"), extras

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
        return action.to("cpu"), extras

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
            action, value, extras = self._get_action_and_value(
                state, lstm_state=lstm_state, done=done
            )
        self._set_train_mode()
        return action.to("cpu"), value.to("cpu"), extras

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
            Tuple[torch.Tensor, Dict[str, Any]]: The action and additional information.
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
            Tuple[torch.Tensor, Dict[str, Any]]: The value and additional information.
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
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: The action, value, and additional information.
        """
        raise NotImplementedError

    def _prepare_state(self, state: Array) -> torch.Tensor:
        """
        Prepare the state for evaluation.

        Args:
            state (Array): The current state.

        Returns:
            torch.Tensor: The prepared state.
        """
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
        pass

    def _set_train_mode(self):
        """
        Set the policy to training mode.
        """
        pass


class CleanRLDiscreteEvaluator(Evaluator):
    def __init__(self, agent, *args, is_lstm=False, **kwargs):
        super().__init__(agent, *args, **kwargs)
        self.is_lstm = is_lstm or hasattr(agent, "lstm")

    def _get_value(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            self._check_inputs(lstm_state, done)
            value = self.agent.get_value(state, lstm_state, done)
        else:
            value = self.agent.get_value(state)
        return value, {}

    def _get_action(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            self._check_inputs(lstm_state, done)
            action = self.agent.get_action(state, lstm_state, done)
        else:
            action = self.agent.get_action(state)
        return action, {}

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            self._check_inputs(lstm_state, done)
            action, log_probs, entropy, value, lstm_state = (
                self.agent.get_action_and_value(state, lstm_state, done)
            )
            return (
                action,
                value,
                {"log_probs": log_probs, "entropy": entropy,
                    "lstm_state": lstm_state},
            )
        else:
            action, log_probs, entropy, value = self.agent.get_action_and_value(
                state)
            return action, value, {"log_probs": log_probs, "entropy": entropy}

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

    def _check_inputs(self, lstm_state, done):
        assert (
            lstm_state is not None
        ), "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
        assert (
            done is not None
        ), "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
        return True


class MoolibEvaluator(Evaluator):
    def __init__(self, agent, *args, **kwargs):
        super().__init__(agent, *args, **kwargs)
        # Make cpu copy of model
        original_device = "cuda"
        agent.to(self.device)
        self.agent = copy.deepcopy(agent)
        agent.to(original_device)

    def _update_agent(self):
        self.agent.load_state_dict(self._agent_reference.state_dict())

    def _get_value(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        state["done"] = done
        output, lstm_state = self.agent(state, lstm_state)
        value = output["baseline"].reshape(-1, 1)
        return value, {}

    def _get_action(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        state["done"] = done
        output, lstm_state = self.agent(state, lstm_state)
        action = output["action"]
        return action, {}

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        self._check_inputs(lstm_state, done)
        state["done"] = done
        output, lstm_state = self.agent(state, lstm_state)
        action = output["action"]
        value = output["baseline"].reshape(-1, 1)
        return (action, value, {"lstm_state": lstm_state})

    def _check_inputs(self, lstm_state, done):
        assert (
            lstm_state is not None
        ), "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
        return True

    def _prepare_state(self, state) -> torch.Tensor:
        full_dict = defaultdict(list)
        for obs_dict in state:
            for k, v in obs_dict.items():
                full_dict[k].append(v)
        tensor_dict = {key: torch.unsqueeze(torch.Tensor(np.stack(val_list)), 0).to(self.device)
                       for key, val_list in full_dict.items()}
        return tensor_dict

    def _set_eval_mode(self):
        self.agent.eval()

    def _set_train_mode(self):
        self.agent.train()
