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
        self.agent = agent
        self.device = device
        self.preprocess_obs = preprocess_obs

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
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)

        with torch.no_grad():
            value, lstm_state, extras = self._get_value(
                state, lstm_state=lstm_state, done=done
            )
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
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)

        with torch.no_grad():
            action, lstm_state, extras = self._get_action(
                state, lstm_state=lstm_state, done=done
            )
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
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)

        with torch.no_grad():
            action, value, extras = self._get_action_and_value(
                state, lstm_state=lstm_state, done=done
            )
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
        if self.preprocess_obs is not None:
            state = self.preprocess_obs(state)
        state = torch.Tensor(state)
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
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: The prepared LSTM state and done flag.
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


class CleanRLDiscreteEvaluator(Evaluator):
    def __init__(self, agent, is_lstm=False, *args, **kwargs):
        super().__init__(agent, *args, **kwargs)
        self.agent = agent
        self.is_lstm = is_lstm or hasattr(agent, "lstm")

    def _get_value(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            assert (
                lstm_state is not None
            ), "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            assert (
                done is not None
            ), "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            value = self.agent.get_value(state, lstm_state, done)
        else:
            value = self.agent.get_value(state)
        return value, {}

    def _get_action(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            assert (
                lstm_state is not None
            ), "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            assert (
                done is not None
            ), "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            action = self.agent.get_action(state, lstm_state, done)
        else:
            action = self.agent.get_action(state)
        return action, {}

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            assert (
                lstm_state is not None
            ), "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            assert (
                done is not None
            ), "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
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
