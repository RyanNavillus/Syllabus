import warnings
import torch
from torch.distributions.categorical import Categorical

from syllabus.core.utils import UsageError


# TODO: Document this class
class Evaluator:
    def __init__(self, agent, device=None, preprocess_obs=None):
        self.agent = agent
        self.device = device
        self.preprocess_obs = preprocess_obs

    def get_value(self, state, lstm_state=None, done=None):
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)

        with torch.no_grad():
            value, lstm_state, extras = self._get_value(state, lstm_state=lstm_state, done=done)
        return value.to("cpu"), extras

    def get_action(self, state, lstm_state=None, done=None):
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)

        with torch.no_grad():
            action, lstm_state, extras = self._get_action(state, lstm_state=lstm_state, done=done)
        return action.to("cpu"), extras

    def get_action_and_value(self, state, lstm_state=None, done=None):
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state, done = self._prepare_lstm(lstm_state, done)

        with torch.no_grad():
            action, value, extras = self._get_action_and_value(state, lstm_state=lstm_state, done=done)
        return action.to("cpu"), value.to("cpu"), extras

    def _get_value(self, state, lstm_state=None, done=None):
        raise NotImplementedError

    def _get_action(self, state, lstm_state=None, done=None):
        raise NotImplementedError

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        raise NotImplementedError

    def _prepare_state(self, state):
        if self.preprocess_obs is not None:
            state = self.preprocess_obs(state)
        state = torch.Tensor(state)
        if self.device is not None:
            state = state.to(self.device)
        return state

    def _prepare_lstm(self, lstm_state, done):
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
            assert lstm_state is not None, "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            assert done is not None, "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            value = self.agent.get_value(state, lstm_state, done)
        else:
            value = self.agent.get_value(state)
        return value, {}

    def _get_action(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            assert lstm_state is not None, "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            assert done is not None, "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            action = self.agent.get_action(state, lstm_state, done)
        else:
            action = self.agent.get_action(state)
        return action, {}

    def _get_action_and_value(self, state, lstm_state=None, done=None):
        if self.is_lstm:
            assert lstm_state is not None, "LSTM state must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            assert done is not None, "Done must be provided. Make sure to configure any LSTM-specific settings for your curriculum."
            action, log_probs, entropy, value, lstm_state = self.agent.get_action_and_value(state, lstm_state, done)
            return action, value, {"log_probs": log_probs, "entropy": entropy, "lstm_state": lstm_state}
        else:
            action, log_probs, entropy, value = self.agent.get_action_and_value(state)
            return action, value, {"log_probs": log_probs, "entropy": entropy}
