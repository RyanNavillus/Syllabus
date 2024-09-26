import torch
from torch.distributions.categorical import Categorical


# TODO: Document this class
class Evaluator:
    def __init__(self, agent, device=None, preprocess_obs=None):
        self.agent = agent
        self.device = device
        self.preprocess_obs = preprocess_obs

    def get_value(self, state, lstm_state=None, done=None):
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state = self._prepare_lstm_state(lstm_state)

        with torch.no_grad():
            value, lstm_state, extras = self._get_value(state, lstm_state=lstm_state, done=done)
            return value.to("cpu"), lstm_state, extras

    def get_action(self, state, lstm_state=None, done=None):
        state = self._prepare_state(state)
        if lstm_state is not None:
            lstm_state = self._prepare_lstm_state(lstm_state)

        with torch.no_grad():
            action, lstm_state, extras = self._get_action(state, lstm_state=lstm_state, done=done)
            return action.to("cpu"), lstm_state, extras

    def _get_value(self, state, lstm_state=None, done=None):
        raise NotImplementedError

    def _get_action(self, state, lstm_state=None, done=None):
        raise NotImplementedError

    def _prepare_state(self, state):
        if self.preprocess_obs is not None:
            state = self.preprocess_obs(state)
        state = torch.Tensor(state)
        if self.device is not None:
            state = state.to(self.device)
        return state

    def _prepare_lstm_state(self, lstm_state):
        lstm_state = torch.Tensor(lstm_state)
        if self.device is not None:
            lstm_state = lstm_state.to(self.device)
        return lstm_state

    def get_action_value(self, state, lstm_state=None):
        return self.get_action(state, lstm_state=lstm_state), self.get_value(state, lstm_state=lstm_state)


class CleanRLDiscreteEvaluator(Evaluator):
    def __init__(self, agent, *args, **kwargs):
        super().__init__(agent, *args, **kwargs)
        self.agent = agent

    def _get_value(self, state, lstm_state=None, done=None):
        if lstm_state is not None and done is not None:     # For LSTM models
            hidden, lstm_state = self.agent.get_states(state, lstm_state, done)
        else:   # For non-LSTM models
            hidden = state
        return self.agent.critic(hidden), lstm_state, {}

    def _get_action(self, state, lstm_state=None, done=None):
        if lstm_state is not None and done is not None:     # For LSTM models
            hidden, lstm_state = self.agent.get_states(state, lstm_state, done)
        else:   # For non-LSTM models
            hidden = state
        logits = self.agent.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.sample(), lstm_state, {}
