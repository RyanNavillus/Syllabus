import warnings
from typing import Any, List, Union

import gymnasium as gym
import numpy as np
import torch

from syllabus.core import Curriculum
from syllabus.core.evaluator import Evaluator
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace
from syllabus.utils import UsageError

from .task_sampler import TaskSampler


class RolloutStorage(object):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        requires_value_buffers: bool,
        observation_space: gym.Space,   # TODO: Use np array when space is box or discrete
        num_minibatches: int = 1,
        buffer_size: int = 2,
        action_space: gym.Space = None,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        lstm_size: int = None,
        evaluator: Evaluator = None,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        # Hack to prevent overflow from lagging updates.
        self.buffer_steps = num_steps * buffer_size
        self.num_processes = num_processes
        self._requires_value_buffers = requires_value_buffers
        self.num_minibatches = num_minibatches
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self.evaluator = evaluator
        self.device = device

        if self.num_processes % self.num_minibatches != 0:
            raise UsageError(
                f"Number of processes {self.num_processes} must be divisible by the number of minibatches {self.num_minibatches}."
            )

        self.tasks = torch.zeros(self.buffer_steps, num_processes, 1, dtype=torch.int)
        self.masks = torch.ones(self.buffer_steps + 1, num_processes, 1)

        self.lstm_states = None
        if lstm_size is not None:
            self.lstm_states = (
                torch.zeros(self.buffer_steps + 1, num_processes, lstm_size),
                torch.zeros(self.buffer_steps + 1, num_processes, lstm_size),
            )

        self.obs = {env_idx: [None for _ in range(self.buffer_steps)] for env_idx in range(self.num_processes)}
        self.env_steps = [0] * num_processes
        self.value_steps = torch.zeros(num_processes, dtype=torch.int)

        if requires_value_buffers:
            self.returns = torch.zeros(self.buffer_steps + 1, num_processes, 1)
            self.rewards = torch.zeros(self.buffer_steps, num_processes, 1)
            self.value_preds = torch.zeros(self.buffer_steps + 1, num_processes, 1)
        else:
            if action_space is None:
                raise ValueError(
                    "Action space must be provided to PLR for strategies 'policy_entropy', 'least_confidence', 'min_margin'"
                )
            self.action_log_dist = torch.zeros(self.buffer_steps, num_processes, action_space.n)

        self.num_steps = num_steps
        self.env_to_idx = {}
        self.max_idx = 0
        self.to(self.device)

    @property
    def using_lstm(self):
        return self.lstm_states is not None

    def to(self, device):
        self.device = device
        self.masks = self.masks.to(device)
        self.tasks = self.tasks.to(device)

        if self.using_lstm:
            self.lstm_states = (
                self.lstm_states[0].to(device),
                self.lstm_states[1].to(device),
            )
        if self._requires_value_buffers:
            self.rewards = self.rewards.to(device)
            self.value_preds = self.value_preds.to(device)
            self.returns = self.returns.to(device)
        else:
            self.action_log_dist = self.action_log_dist.to(device)

    def get_index(self, env_index):
        """ Map the environment ids to indices in the buffer. """
        if env_index not in self.env_to_idx:
            assert self.max_idx < self.num_processes, f"Number of environments {self.max_idx} exceeds num_processes {self.num_processes}."
            self.env_to_idx[env_index] = self.max_idx
            self.max_idx += 1
        return self.env_to_idx[env_index]

    def insert_at_index(self, env_index, mask, obs=None, reward=None, task=None, steps=1):
        assert steps < self.buffer_steps, f"Number of steps {steps} exceeds buffer size {self.buffer_steps}. Increase PLR's num_steps or decrease environment wrapper's batch size."
        env_index = self.get_index(env_index)
        step = self.env_steps[env_index]
        end_step = step + steps
        assert end_step < self.buffer_steps, f"Number of insert of {steps} steps at {step} exceeds buffer size {self.buffer_steps}. Increase PLR's num_steps or decrease environment wrapper's batch size."
        self.masks[step + 1:end_step + 1, env_index].copy_(torch.as_tensor(mask[:, None]))

        if obs is not None:
            self.obs[env_index][step: end_step] = obs

        if reward is not None:
            self.rewards[step:end_step, env_index].copy_(torch.as_tensor(reward[:, None]))

        # if action_log_dist is not None:
        #     self.action_log_dist[step:end_step, env_index].copy_(torch.as_tensor(action_log_dist[:, None]))

        if task is not None:
            try:
                int(task[0])
            except TypeError:
                assert isinstance(
                    task, int), f"Provided task must be an integer, got {task[0]} with type {type(task[0])} instead."
            self.tasks[step:end_step, env_index].copy_(torch.as_tensor(np.array(task)[:, None]))

        self.env_steps[env_index] += steps

        # Get value predictions if batch is ready
        value_steps = self.value_steps.numpy()
        while all((self.env_steps - value_steps) > 0):
            self.get_value_predictions()

        # Check if the buffer is ready to be updated. Wait until we have enough value predictions.
        if self.value_steps[env_index] >= self.num_steps + 1:
            if self._requires_value_buffers:
                self.compute_returns(self._gamma, self._gae_lambda, env_index)
            return env_index
        return None

    def get_value_predictions(self):
        value_steps = self.value_steps.numpy()
        process_chunks = np.split(np.arange(self.num_processes), self.num_minibatches)

        for processes in process_chunks:
            obs = [self.obs[env_idx][value_steps[env_idx]] for env_idx in processes]
            lstm_states = dones = None
            if self.using_lstm:
                lstm_states = (
                    torch.unsqueeze(self.lstm_states[0][value_steps[processes], processes], 0),
                    torch.unsqueeze(self.lstm_states[1][value_steps[processes], processes], 0),
                )
                dones = torch.squeeze(1 - self.masks[value_steps[processes], processes], -1).int()

            # Get value predictions and check for common usage errors
            try:
                values, lstm_states, _ = self.evaluator.get_value(obs, lstm_states, dones)
            except RuntimeError as e:
                raise UsageError(
                    "Encountered an error getting values for PLR. Check that lstm_size is set correctly and that there are no errors in the evaluator's get_value implementation."
                ) from e

            self.value_preds[value_steps[processes], processes] = values.to(self.device)
            self.value_steps[processes] += 1   # Increase index to store lstm_states and next iteration
            value_steps = self.value_steps.numpy()

            if self.using_lstm:
                assert lstm_states is not None, "Evaluator must return lstm_state in extras for PLR."

                # Place new lstm_states in next step
                self.lstm_states[0][value_steps[processes], processes] = lstm_states[0].to(self.lstm_states[0].device)
                self.lstm_states[1][value_steps[processes], processes] = lstm_states[1].to(self.lstm_states[1].device)

    def after_update(self, env_index):
        # After consuming the first num_steps of data, remove them and shift the remaining data in the buffer
        self.tasks[:, env_index] = self.tasks[:, env_index].roll(-self.num_steps, 0)
        self.masks[:, env_index] = self.masks[:, env_index].roll(-self.num_steps, 0)
        self.obs[env_index] = self.obs[env_index][self.num_steps:]

        if self.using_lstm:
            self.lstm_states[0][:, env_index] = self.lstm_states[0][:, env_index].roll(-self.num_steps, 0)
            self.lstm_states[1][:, env_index] = self.lstm_states[1][:, env_index].roll(-self.num_steps, 0)

        if self._requires_value_buffers:
            self.returns[:, env_index] = self.returns[:, env_index].roll(-self.num_steps, 0)
            self.rewards[:, env_index] = self.rewards[:, env_index].roll(-self.num_steps, 0)
            self.value_preds[:, env_index] = self.value_preds[:, env_index].roll(-(self.num_steps), 0)
        else:
            self.action_log_dist[:, env_index] = self.action_log_dist[:, env_index].roll(-self.num_steps, 0)

        self.env_steps[env_index] -= self.num_steps
        self.value_steps[env_index] -= self.num_steps

    def compute_returns(self, gamma, gae_lambda, env_index):
        assert self._requires_value_buffers, "Selected strategy does not use compute_rewards."
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = (
                self.rewards[step, env_index]
                + gamma * self.value_preds[step + 1, env_index] * self.masks[step + 1, env_index]
                - self.value_preds[step, env_index]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1, env_index] * gae
            self.returns[step, env_index] = gae + self.value_preds[step, env_index]


class PrioritizedLevelReplay(Curriculum):
    """ Prioritized Level Replay (PLR) Curriculum.

    Args:
        task_space (TaskSpace): The task space to use for the curriculum.
        *curriculum_args: Positional arguments to pass to the curriculum.
        task_sampler_kwargs_dict (dict): Keyword arguments to pass to the task sampler. See TaskSampler for details.
        action_space (gym.Space): The action space to use for the curriculum. Required for some strategies.
        device (str): The device to use to store curriculum data, either "cpu" or "cuda".
        num_steps (int): The number of steps to store in the rollouts.
        num_processes (int): The number of parallel environments.
        gamma (float): The discount factor used to compute returns
        gae_lambda (float): The GAE lambda value.
        suppress_usage_warnings (bool): Whether to suppress warnings about improper usage.
        **curriculum_kwargs: Keyword arguments to pass to the curriculum.
    """
    REQUIRES_STEP_UPDATES = True
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(
        self,
        task_space: Union[DiscreteTaskSpace, MultiDiscreteTaskSpace],
        observation_space: gym.Space,
        *curriculum_args,
        task_sampler_kwargs_dict: dict = None,
        action_space: gym.Space = None,
        lstm_size: int = None,
        device: str = "cpu",
        num_steps: int = 256,
        num_processes: int = 64,
        num_minibatches: int = 1,
        buffer_size: int = 4,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        suppress_usage_warnings=False,
        evaluator: Evaluator = None,
        **curriculum_kwargs,
    ):
        # Preprocess curriculum intialization args
        if task_sampler_kwargs_dict is None:
            task_sampler_kwargs_dict = {}

        self._strategy = task_sampler_kwargs_dict.get("strategy", None)
        if not isinstance(task_space, (DiscreteTaskSpace, MultiDiscreteTaskSpace)):
            raise ValueError(
                f"Task space must be discrete or multi-discrete, got {task_space}."
            )
        if "num_actors" in task_sampler_kwargs_dict and task_sampler_kwargs_dict['num_actors'] != num_processes:
            warnings.warn(
                f"Overwriting 'num_actors' {task_sampler_kwargs_dict['num_actors']} in task sampler kwargs with PLR num_processes {num_processes}.", stacklevel=2)
        task_sampler_kwargs_dict["num_actors"] = num_processes
        super().__init__(task_space, *curriculum_args, **curriculum_kwargs)

        # Number of steps stored in rollouts and used to update task sampler
        self._num_steps = num_steps
        self._num_processes = num_processes  # Number of parallel environments
        self._supress_usage_warnings = suppress_usage_warnings
        self.evaluator = evaluator
        self._task2index = {task: i for i, task in enumerate(self.tasks)}

        self._task_sampler = TaskSampler(self.tasks, self._num_steps,
                                         action_space=action_space, **task_sampler_kwargs_dict)
        self._rollouts = RolloutStorage(
            self._num_steps,
            self._num_processes,
            self._task_sampler.requires_value_buffers,
            observation_space,
            num_minibatches=num_minibatches,
            buffer_size=buffer_size,
            action_space=action_space,
            gamma=gamma,
            gae_lambda=gae_lambda,
            lstm_size=lstm_size,
            evaluator=evaluator,
            device=device,
        )
        self._rollouts.to(device)

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        return self._task_sampler.sample_weights()

    def sample(self, k: int = 1) -> Union[List, Any]:
        if self._should_use_startup_sampling():
            return self._startup_sample()
        else:
            return [self._task_sampler.sample() for _ in range(k)]

    def update_on_step(self, task, obs, rew, term, trunc, info, progress, env_id: int = None) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        assert env_id is not None, "env_id must be provided for PLR updates."

        # Update rollouts
        update_id = self._rollouts.insert_at_index(
            env_id,
            mask=np.array([not (term or trunc)]),
            reward=np.array([rew]),
            obs=np.array([obs]),
        )

        # Update task sampler
        if update_id is not None:
            self._update_sampler(update_id)

    def update_on_step_batch(self, step_results, env_id=None) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        assert env_id is not None, "env_id must be provided for PLR updates."

        tasks, obs, rews, terms, truncs, _, _ = step_results
        update_id = self._rollouts.insert_at_index(
            env_id,
            mask=np.logical_not(np.logical_or(terms, truncs)),
            reward=rews,
            obs=obs,
            steps=len(rews),
            task=tasks,
        )

        # Update task sampler
        if update_id is not None:
            self._update_sampler(update_id)

    def _update_sampler(self, env_id):
        """ Update the task sampler with the current rollouts. """
        self._task_sampler.update_with_rollouts(self._rollouts, env_id)
        self._rollouts.after_update(env_id)
        self._task_sampler.after_update()

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """
        Log the task distribution to the provided tensorboard writer.
        """
        logs = [] if logs is None else logs
        metrics = self._task_sampler.metrics()
        logs.append(("curriculum/proportion_seen", metrics["proportion_seen"]))
        logs.append(("curriculum/score", metrics["score"]))
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)
