import torch
from torch.distributions.categorical import Categorical


class Evaluator:
    def __init__(self, agent, get_value, get_action, device=None, preprocess_obs=None):
        self.agent = agent
        self._get_value = get_value
        self._get_action = get_action
        self.device = device
        self.preprocess_obs = preprocess_obs

    def get_value(self, state):
        assert self._get_value is not None, "get_value is not implemented"
        state = self._prepare_state(state)
        return self._get_value(state)

    def get_action(self, state):
        assert self._get_action is not None, "get_action is not implemented"
        state = self._prepare_state(state)
        return self._get_action(state)

    def _prepare_state(self, state):
        if self.preprocess_obs is not None:
            state = self.preprocess_obs(state)
        state = torch.Tensor(state)
        if self.device is not None:
            state = state.to(self.device)
        return state

    def get_action_value(self, state):
        return self.get_action(state), self.get_value(state)


class CleanRLDiscreteEvaluator(Evaluator):
    def __init__(self, agent, *args, **kwargs):
        super().__init__(agent, self._get_value, self._get_action, *args, **kwargs)
        self.agent = agent

    def _get_value(self, state):
        return self.agent.critic(state)

    def _get_action(self, state):
        logits = self.agent.actor(state)
        probs = Categorical(logits=logits)
        return probs.sample()
