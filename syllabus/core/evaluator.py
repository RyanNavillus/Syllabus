import torch
from torch.distributions.categorical import Categorical


class Evaluator:
    def __init__(self, agent, device=None, preprocess_obs=None):
        self.agent = agent
        self.device = device
        self.preprocess_obs = preprocess_obs

    def get_value(self, state):
        state = self._prepare_state(state)
        with torch.no_grad():
            return self._get_value(state).to("cpu")

    def get_action(self, state):
        state = self._prepare_state(state)
        with torch.no_grad():
            return self._get_action(state).to("cpu")

    def _get_value(self, state):
        raise NotImplementedError

    def _get_action(self, state):
        raise NotImplementedError

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
        super().__init__(agent, *args, **kwargs)
        self.agent = agent

    def _get_value(self, state):
        return self.agent.critic(state)

    def _get_action(self, state):
        logits = self.agent.actor(state)
        probs = Categorical(logits=logits)
        return probs.sample()
