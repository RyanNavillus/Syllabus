from typing import Any, Callable, Dict, Optional, Tuple, Union
import warnings

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from syllabus.core.utils import UsageError

Array = Union[np.ndarray, Tensor]
LSTMState = Tuple[Array, Array]

# TODO: Add deterministic eval support
# TODO: Implement norm and reward normalization


class Evaluator:
    """An interface for evaluating a trained agent, used by several curricula."""

    def __init__(
        self,
        agent: Any,
        make_eval_env: Callable = None,
        device: Optional[torch.device] = None,
        preprocess_obs: Optional[Callable] = None,
        num_eval_processes: int = 1,
        norm_obs: bool = False,
        norm_reward: bool = False,
    ):
        """
        Initialize the Evaluator.

        Args:
            agent (Any): The trained agent to be evaluated.
            device (Optional[torch.device]): The device to run the evaluation on.
            preprocess_obs (Optional[Any]): A function to preprocess observations.
        """
        self.agent = agent
        self.make_eval_env = make_eval_env
        self.device = device
        self.preprocess_obs = preprocess_obs
        self.num_eval_processes = num_eval_processes
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward

        if self.make_eval_env is not None and not self._check_envs(make_eval_env):
            warnings.warn("The provided make_eval_env is not valid.")

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

        self._set_eval_mode()
        with torch.no_grad():
            value, lstm_state, extras = self._get_value(
                state, lstm_state=lstm_state, done=done
            )
        self._set_train_mode()

        return value.to("cpu"), lstm_state, extras

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

        self._set_eval_mode()
        with torch.no_grad():
            action, lstm_state, extras = self._get_action(
                state, lstm_state=lstm_state, done=done
            )
        self._set_train_mode()

        return action.to("cpu"), lstm_state, extras

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

        self._set_eval_mode()
        with torch.no_grad():
            action, value, extras = self._get_action_and_value(
                state, lstm_state=lstm_state, done=done
            )
        self._set_train_mode()
        return action.to("cpu"), value.to("cpu"), lstm_state, extras

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

    def evaluate_agent(self, fast_biased: bool = False, full_data: bool = False, num_episodes=10, num_steps=None, eval_envs=None):
        """
        Evaluate the agent on the environment.
        The fast_biased option will use the first num_episodes completed episodes, which biases towards shorter episodes.
        This may increase or decrease the expected return depending on the environment's reward function.

        Args:
            fast_biased (bool): Whether to use a faster, biased evaluation.
            full_data (bool): Whether to return the full evaluation data.

        Returns:
            Dict[str, Any]: The evaluation data.
        """
        # TODO: Run for specific number of steps and continue with provided rollout storage
        # TODO: Check that eval_envs is usable and give warning. Throw error here for same check
        # TODO: Test with nonmatching num_episodes and env size. Make it work for any combo.
        eval_envs = eval_envs if eval_envs is not None else self.create_eval_envs()
        assert self._check_envs(eval_envs)
        self._set_eval_mode()
        obs, _ = eval_envs.reset()
        assert obs.shape[0] == num_episodes, "Number of episodes must match the batch size of the environment."

        episode_rewards = -np.ones(num_episodes)
        completed = np.zeros(num_episodes)
        fast_index = 0

        while sum(completed) < num_episodes:
            action, _, _ = self.get_action_and_value(torch.Tensor(obs).to(self.device))

            obs, _, _, _, infos = eval_envs.step(action.cpu().numpy())
            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        if fast_biased:
                            episode_rewards[fast_index] = info['episode']['r']
                            fast_index += 1
                        elif completed[i] == 0:
                            episode_rewards[i] = info['episode']['r']
                            completed[i] = 1

        self._set_train_mode()
        return episode_rewards

    def create_eval_envs(self):
        return gym.vector.SyncVectorEnv([self.make_eval_env for _ in range(self.num_eval_processes)])

    def _check_envs(self, make_eval_env):
        eval_env = make_eval_env()
        return eval_env is not None and hasattr(eval_env, "reset") and hasattr(eval_env, "step")


class CleanRLDiscreteEvaluator(Evaluator):
    def __init__(self, agent, *args, is_lstm=False, **kwargs):
        super().__init__(agent, *args, **kwargs)
        self.agent = agent
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
