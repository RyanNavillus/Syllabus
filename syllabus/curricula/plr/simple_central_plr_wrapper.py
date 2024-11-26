import warnings
from typing import Any, Dict, List, Union

import gymnasium as gym
import torch

from syllabus.core import Curriculum
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace

from .task_sampler import TaskSampler


class RolloutStorage(object):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
    ):
        self.tasks = torch.zeros(num_steps, num_processes, dtype=torch.int)
        self.masks = torch.ones(num_steps, num_processes, dtype=torch.int)
        self.scores = torch.zeros(num_steps, num_processes)

        self.num_processes = num_processes
        self.actor_steps = torch.zeros(num_processes, dtype=torch.int)
        self.actors = set()

    def to(self, device):
        self.tasks = self.tasks.to(device)
        self.masks = self.masks.to(device)
        self.scores = self.scores.to(device)

    def insert(self, tasks, masks, scores, actors):
        steps = tasks.shape[0]
        for step in range(steps):
            self.tasks[self.actor_steps[actors] + step, actors] = tasks.int().cpu()[step]
            self.masks[self.actor_steps[actors] + step, actors] = masks.cpu()[step]
            self.scores[self.actor_steps[actors] + step, actors] = scores.cpu()[step]
        self.actor_steps[actors] += steps
        self.actors.update(actors)

    def after_update(self):
        self.masks[0].copy_(self.masks[-1])
        self.actor_steps = torch.zeros(self.num_processes, dtype=torch.int)

        self.actors = set()

    def ready(self):
        return len(self.actors) == self.num_processes


class SimpleCentralizedPrioritizedLevelReplay(Curriculum):
    """ Prioritized Level Replay (PLR) Curriculum.

    Args:
        task_space (TaskSpace): The task space to use for the curriculum.
        *curriculum_args: Positional arguments to pass to the curriculum.
        task_sampler_kwargs_dict (dict): Keyword arguments to pass to the task sampler. See TaskSampler for details.
        action_space (gym.Space): The action space to use for the curriculum. Required for some strategies.
        device (str): The device to use to store curriculum data, either "cpu" or "cuda".
        num_steps (int): The number of steps to store in the rollouts.
        num_processes (int): The number of parallel environments.
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
        suppress_usage_warnings=False,
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

        self._num_steps = num_steps  # Number of steps stored in rollouts and used to update task sampler
        self._num_processes = num_processes  # Number of parallel environments
        self._supress_usage_warnings = suppress_usage_warnings
        self._task2index = {task: i for i, task in enumerate(self.tasks)}
        self._task_sampler = TaskSampler(self.tasks, self._num_steps,
                                         action_space=action_space, **task_sampler_kwargs_dict)
        self._rollouts = RolloutStorage(
            self._num_steps,
            self._num_processes,
        )
        self._rollouts.to(device)

        # TODO: Fix this feature
        self.num_updates = 0  # Used to ensure proper usage
        self.num_samples = 0  # Used to ensure proper usage

    def update(self, metrics: Dict):
        """
        Update the curriculum with arbitrary inputs.
        """
        self.num_updates += 1
        tasks = metrics["tasks"]
        scores = metrics["scores"]
        actors = metrics["actors"]
        masks = torch.Tensor(1 - metrics["dones"].int())

        # Update rollouts
        self._rollouts.insert(tasks, masks, scores, actors)

        # Update task sampler
        if self._rollouts.ready():
            self._task_sampler._update_with_scores(self._rollouts)
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
            logs.append((f"curriculum/{name}_staleness", metrics["task_staleness"][idx]))
            logs.append((f"curriculum/{name}_score", metrics["task_scores"][idx]))
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)
