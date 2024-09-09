import torch
from torch.distributions.categorical import Categorical


class Evaluator:
    def __init__(self, agent, get_value=None, get_action=None):
        self.agent = agent
        self._get_value = get_value
        self._get_action = get_action

    def get_value(self, state):
        assert self._get_value is not None, "get_value is not defined. Provide a get_value function to the Evaluator."
        return self._get_value(state)

    def get_action(self, state):
        assert self._get_action is not None, "get_action is not defined. Provide a get_action function to the Evaluator."
        return self._get_action(state)

    def get_action_value(self, state):
        return self.get_action(state), self.get_value(state)


class CleanRLDiscreteEvaluator(Evaluator):
    def __init__(self, agent, device=None, preprocess_obs=None):
        super().__init__(agent)
        self.agent = agent
        self.device = device
        self.preprocess_obs = preprocess_obs

    def get_value(self, state):
        if self.preprocess_obs is not None:
            state = self.preprocess_obs([state])
        state = torch.Tensor(state)
        if self.device is not None:
            state = state.to(self.device)

        return self.agent.critic(state)

    def get_action(self, state):
        if self.preprocess_obs is not None:
            state = self.preprocess_obs([state])
        state = torch.Tensor(state)
        if self.device is not None:
            state = state.to(self.device)

        logits = self.agent.actor(state)
        probs = Categorical(logits=logits)
        return probs.sample()
