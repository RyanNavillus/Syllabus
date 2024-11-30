import warnings
from typing import Any, Dict, List, Tuple, Union

import gymnasium as gym
import torch

from syllabus.core import Curriculum
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace
from syllabus.utils import UsageError

from .task_sampler import TaskSampler


class RolloutStorage():
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        requires_value_buffers: bool,
        action_space: gym.Space = None,
    ):
        self.num_processes = num_processes
        self._requires_value_buffers = requires_value_buffers
        self.tasks = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        self.masks = torch.ones(num_steps + 1, num_processes, 1, dtype=torch.int)

        if requires_value_buffers:
            self.returns = torch.zeros(num_steps + 1, num_processes, 1)
            self.rewards = torch.zeros(num_steps, num_processes, 1)
            self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        else:
            if action_space is None:
                raise ValueError(
                    "Action space must be provided to PLR for strategies 'policy_entropy', 'least_confidence', 'min_margin'"
                )
            self.action_log_dist = torch.zeros(num_steps, num_processes, action_space.n)

        self.num_steps = num_steps
        self.env_steps = torch.zeros(num_processes, dtype=torch.int)
        self.env_to_idx = {}
        self.max_idx = 0

        # Logging
        self.final_return_mean = 0.0
        self.first_value_mean = 0.0

    def get_idxs(self, env_ids):
        """ Map the environment ids to indices in the buffer. """
        idxs = []
        for env_id in env_ids:
            if env_id not in self.env_to_idx:
                self.env_to_idx[env_id] = self.max_idx
                self.max_idx += 1
            idxs.append(self.env_to_idx[env_id])
        return idxs

    def to(self, device):
        self.masks = self.masks.to(device)
        self.tasks = self.tasks.to(device)
        if self._requires_value_buffers:
            self.rewards = self.rewards.to(device)
            self.value_preds = self.value_preds.to(device)
            self.returns = self.returns.to(device)
        else:
            self.action_log_dist = self.action_log_dist.to(device)

    def insert(self, masks, action_log_dist=None, value_preds=None, rewards=None, tasks=None, next_values=None, env_ids=None):
        # Convert env_ids to indices in the buffer
        if env_ids is None:
            env_ids = list(range(self.num_processes))
        env_idxs = self.get_idxs(env_ids)

        if self._requires_value_buffers:
            assert (value_preds is not None and rewards is not None), "Selected strategy requires value_preds and rewards"
            if len(rewards.shape) == 3:
                rewards = rewards.squeeze(2)
            self.value_preds[self.env_steps[env_idxs], env_idxs] = torch.as_tensor(
                value_preds).reshape((len(env_idxs), 1)).cpu()
            if next_values is not None:
                self.value_preds[self.env_steps[env_idxs] + 1,
                                 env_idxs] = torch.as_tensor(next_values).reshape((len(env_idxs), 1)).cpu()
            self.rewards[self.env_steps[env_idxs], env_idxs] = torch.as_tensor(
                rewards).reshape((len(env_idxs), 1)).cpu()
            self.masks[self.env_steps[env_idxs] + 1,
                       env_idxs] = torch.IntTensor(masks.cpu()).reshape((len(env_idxs), 1))
        else:
            self.action_log_dist[self.env_steps[env_idxs],
                                 env_idxs] = action_log_dist.reshape((len(env_idxs), -1)).cpu()

        if tasks is not None:
            assert isinstance(tasks[0], int), "Provided task must be an integer"
            self.tasks[self.env_steps[env_idxs], env_idxs] = torch.IntTensor(tasks).reshape((len(env_idxs), 1))
        self.env_steps[env_idxs] = (self.env_steps[env_idxs] + 1) % (self.num_steps + 1)

    def after_update(self):
        env_idxs = (self.env_steps == self.num_steps).nonzero()
        self.env_steps[env_idxs] = (self.env_steps[env_idxs] + 1) % (self.num_steps + 1)
        self.masks[0, env_idxs].copy_(self.masks[-1, env_idxs])

    def compute_returns(self, gamma, gae_lambda):
        env_idxs = (self.env_steps == self.num_steps).nonzero()
        assert self._requires_value_buffers, "Selected strategy does not use compute_rewards."
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step, env_idxs]
                + gamma * self.value_preds[step + 1, env_idxs] * self.masks[step + 1, env_idxs]
                - self.value_preds[step, env_idxs]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1, env_idxs] * gae
            self.returns[step, env_idxs] = gae + self.value_preds[step, env_idxs]


class CentralizedPrioritizedLevelReplay(Curriculum):
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
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = True

    def __init__(
        self,
        task_space: Union[DiscreteTaskSpace, MultiDiscreteTaskSpace],
        *curriculum_args,
        task_sampler_kwargs_dict: dict = None,
        action_space: gym.Space = None,
        device: str = "cpu",
        num_steps: int = 256,
        num_processes: int = 64,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        suppress_usage_warnings=False,
        **curriculum_kwargs,
    ):
        # Preprocess curriculum intialization args
        if task_sampler_kwargs_dict is None:
            task_sampler_kwargs_dict = {}

        self._strategy = task_sampler_kwargs_dict.get("strategy", None)
        if not isinstance(task_space, (DiscreteTaskSpace, MultiDiscreteTaskSpace)):
            raise UsageError(
                f"Task space must be discrete or multi-discrete, got {task_space}."
            )
        if "num_actors" in task_sampler_kwargs_dict and task_sampler_kwargs_dict['num_actors'] != num_processes:
            warnings.warn(
                f"Overwriting 'num_actors' {task_sampler_kwargs_dict['num_actors']} in task sampler kwargs with PLR num_processes {num_processes}.", stacklevel=2)
        task_sampler_kwargs_dict["num_actors"] = num_processes
        super().__init__(task_space, *curriculum_args, **curriculum_kwargs)

        self._num_steps = num_steps  # Number of steps stored in rollouts and used to update task sampler
        self._num_processes = num_processes  # Number of parallel environments
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._supress_usage_warnings = suppress_usage_warnings
        self._task2index = {task: i for i, task in enumerate(self.tasks)}
        self._task_sampler = TaskSampler(self.tasks, self._num_steps,
                                         action_space=action_space, **task_sampler_kwargs_dict)
        self._rollouts = RolloutStorage(
            self._num_steps,
            self._num_processes,
            self._task_sampler.requires_value_buffers,
            action_space=action_space,
        )
        self._rollouts.to(device)
        # TODO: Fix this feature
        self.num_updates = 0  # Used to ensure proper usage
        self.num_samples = 0  # Used to ensure proper usage

    def _validate_metrics(self, metrics: Dict):
        try:
            masks = torch.Tensor(1 - metrics["dones"].int())
            tasks = metrics["tasks"]
            tasks = [self._task2index[t] for t in tasks]
        except KeyError as e:
            raise KeyError(
                "Missing or malformed PLR update. Must include 'masks', and 'tasks', and all tasks must be in the task space"
            ) from e

        # Parse optional update values (required for some strategies)
        value = next_value = rew = action_log_dist = None
        if self._task_sampler.requires_value_buffers:
            if "value" not in metrics or "rew" not in metrics:
                raise KeyError(
                    f"'value' and 'rew' must be provided in every update for the strategy {self._strategy}."
                )
            value = metrics["value"]
            rew = metrics["rew"]
        else:
            try:
                action_log_dist = metrics["action_log_dist"]
            except KeyError as e:
                raise KeyError(
                    f"'action_log_dist' must be provided in every update for the strategy {self._strategy}."
                ) from e

        if self._task_sampler.requires_value_buffers:
            try:
                next_value = metrics["next_value"]
            except KeyError as e:
                raise KeyError(
                    f"'next_value' must be provided in the update every {self.num_steps} steps for the strategy {self._strategy}."
                ) from e

        env_ids = metrics["env_ids"] if "env_ids" in metrics else None
        assert env_ids is None or len(
            env_ids) <= self._num_processes, "Number of env_ids must be less than or equal to num_processes"

        return masks, tasks, value, rew, action_log_dist, next_value, env_ids

    def update(self, metrics: Dict):
        """
        Update the curriculum with arbitrary inputs.
        """
        self.num_updates += 1
        masks, tasks, value, rew, action_log_dist, next_value, env_ids = self._validate_metrics(metrics)

        # Update rollouts
        self._rollouts.insert(
            masks,
            action_log_dist=action_log_dist,
            value_preds=value,
            rewards=rew,
            tasks=tasks,
            env_ids=env_ids,
            next_values=next_value,
        )

        # Update task sampler
        if any(self._rollouts.env_steps == self._rollouts.num_steps):
            env_idxs = (self._rollouts.env_steps == self._rollouts.num_steps).nonzero()
            if self._task_sampler.requires_value_buffers:
                self._rollouts.compute_returns(self._gamma, self._gae_lambda)
            for idx in env_idxs:
                self._task_sampler.update_with_rollouts(
                    self._rollouts,
                    actor_id=idx,
                )
            self._task_sampler.update_with_rollouts(self._rollouts)
            self._rollouts.after_update()
            self._task_sampler.after_update()

    def _sample_distribution(self) -> List[float]:
        """
        Returns a sample distribution over the task space.
        """
        return self._task_sampler.sample_weights()

    def sample(self, k: int = 1) -> Union[List, Any]:
        self.num_samples += 1
        if self._should_use_startup_sampling():
            return self._startup_sample()
        else:
            return [self._task_sampler.sample() for _ in range(k)]

    def log_metrics(self, writer, logs, step=None, log_n_tasks=1):
        """
        Log the task distribution to the provided tensorboard writer.
        """
        logs = [] if logs is None else logs
        metrics = self._task_sampler.metrics()
        logs.append(("curriculum/proportion_seen", metrics["proportion_seen"]))
        logs.append(("curriculum/score", metrics["score"]))

        tasks = range(self.num_tasks)
        if self.num_tasks > log_n_tasks and log_n_tasks != -1:
            warnings.warn(f"Too many tasks to log {self.num_tasks}. Only logging stats for 1 task.", stacklevel=2)
            tasks = tasks[:log_n_tasks]

        for idx in tasks:
            name = self.task_names(self.tasks[idx], idx)
            logs.append((f"curriculum/{name}_score", metrics["task_scores"][idx]))
            logs.append((f"curriculum/{name}_staleness", metrics["task_staleness"][idx]))
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)
