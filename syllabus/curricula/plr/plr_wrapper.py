from typing import Any, Dict, List, Tuple, Union

import numpy as np
import gym
import torch
from gym.spaces import Discrete, MultiDiscrete
from syllabus.core import Curriculum, UsageError, enumerate_axes
from syllabus.curricula.plr import TaskSampler
from syllabus.task_space import TaskSpace


class RolloutStorage(object):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        requires_value_buffers: bool,
        action_space: gym.Space = None,
    ):
        self._requires_value_buffers = requires_value_buffers
        self.tasks = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

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
        self.step = 0

    def to(self, device):
        self.masks = self.masks.to(device)
        self.tasks = self.tasks.to(device)
        if self._requires_value_buffers:
            self.rewards = self.rewards.to(device)
            self.value_preds = self.value_preds.to(device)
            self.returns = self.returns.to(device)
        else:
            self.action_log_dist = self.action_log_dist.to(device)

    def insert(self, masks, action_log_dist=None, value_preds=None, rewards=None, tasks=None):
        if self._requires_value_buffers:
            assert (value_preds is not None and rewards is not None), "Selected strategy requires value_preds and rewards"
            if len(rewards.shape) == 3:
                rewards = rewards.squeeze(2)
            self.value_preds[self.step].copy_(torch.as_tensor(value_preds))
            self.rewards[self.step].copy_(torch.as_tensor(rewards)[:, None])
            self.masks[self.step + 1].copy_(torch.as_tensor(masks)[:, None])
        else:
            self.action_log_dist[self.step].copy_(action_log_dist)
        if tasks is not None:
            assert isinstance(tasks[0], int), "Provided task must be an integer"
            self.tasks[self.step].copy_(torch.as_tensor(tasks)[:, None])
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        assert self._requires_value_buffers, "Selected strategy does not use compute_rewards."
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step]
                + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]


class PrioritizedLevelReplay(Curriculum):
    REQUIRES_STEP_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = True

    def __init__(
        self,
        task_space: TaskSpace,
        *curriculum_args,
        task_sampler_kwargs_dict: dict = {},
        action_space: gym.Space = None,
        device: str = "cpu",
        num_steps: int = 256,
        num_processes: int = 64,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        suppress_usage_warnings=False,
        **curriculum_kwargs,
    ):
        self._strategy = task_sampler_kwargs_dict.get("strategy", None)
        if not isinstance(task_space.gym_space, Discrete) and not isinstance(task_space.gym_space, MultiDiscrete):
            raise ValueError(
                f"Task space must be discrete or multi-discrete, got {task_space.gym_space}."
            )
        if "num_actors" in task_sampler_kwargs_dict:
            print(f"Overwriting 'num_actors' {task_sampler_kwargs_dict['num_actors']} in task sampler kwargs with PLR num_processes {num_processes}.")

        task_sampler_kwargs_dict["num_actors"] = num_processes
        super().__init__(task_space, *curriculum_args, **curriculum_kwargs)
        self._num_steps = num_steps  # Number of steps stored in rollouts and used to update task sampler
        self._num_processes = num_processes  # Number of parallel environments
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._supress_usage_warnings = suppress_usage_warnings
        # self.tasks = self._enumerate_tasks(task_space)
        self._task2index = {task: i for i, task in enumerate(self.tasks)}
        self._task_sampler = TaskSampler(self.tasks, action_space=action_space, **task_sampler_kwargs_dict)
        self._rollouts = RolloutStorage(
            self._num_steps,
            self._num_processes,
            self._task_sampler.requires_value_buffers,
            action_space=action_space,
        )
        self._rollouts.to(device)
        self.num_updates = 0  # Used to ensure proper usage
        self.num_samples = 0  # Used to ensure proper usage

    def update_on_demand(self, metrics: Dict):
        """
        Update the curriculum with arbitrary inputs.
        """
        self.num_updates += 1
        try:
            masks = metrics["masks"]
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
            if "action_log_dist" not in metrics:
                raise KeyError(
                    f"'action_log_dist' must be provided in every update for the strategy {self._strategy}."
                )
            action_log_dist = metrics["action_log_dist"]

        # Update rollouts
        self._rollouts.insert(
            masks,
            action_log_dist=action_log_dist,
            value_preds=value,
            rewards=rew,
            tasks=tasks,
        )

        # Update task sampler
        # print(self._rollouts.step )
        if self._rollouts.step == 0:
            # print("success")
            if self._task_sampler.requires_value_buffers:
                if "next_value" not in metrics:
                    raise KeyError(
                        "'next_value' must be provided in the update every {self.num_steps} steps for the strategy {self._strategy}."
                    )
                next_value = metrics["next_value"]
                self._rollouts.compute_returns(next_value, self._gamma, self._gae_lambda)
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
        return [self._task_sampler.sample() for k in range(k + 1)]

    def update_on_step(self, obs, rew, done, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        raise NotImplementedError(
            "PrioritizedLevelReplay does not support the step updates. Use on_demand from the learner process."
        )

    def update_on_step_batch(
        self, step_results: List[Tuple[int, int, int, int]]
    ) -> None:
        """
        Update the curriculum with a batch of step results from the environment.
        """
        raise NotImplementedError(
            "PrioritizedLevelReplay does not support the step updates. Use on_demand from the learner process."
        )

    def update_on_episode(self, episode_return: float, trajectory: List = None) -> None:
        """
        Update the curriculum with episode results from the environment.
        """
        raise NotImplementedError(
            "PrioritizedLevelReplay does not support the episode updates. Use on_demand from the learner process."
        )

    def update_task_progress(self, task: Any, success_prob: float) -> None:
        """
        Update the curriculum with a task and its success probability upon
        success or failure.
        """
        if not self._supress_usage_warnings and self.num_updates == 0 and self.num_samples > self._num_processes * 2:
            raise UsageError("PLR has not been updated yet. Please call update_curriculum() in your learner process.")

    def _enumerate_tasks(self, space):
        assert isinstance(space, Discrete) or isinstance(space, MultiDiscrete), f"Unsupported task space {space}: Expected Discrete or MultiDiscrete"
        if isinstance(space, Discrete):
            return list(range(space.n))
        else:
            return list(enumerate_axes(space.nvec))

    def log_metrics(self, writer, step=None):
        """Log the task distribution to the provided tensorboard writer.

        """
        super().log_metrics(writer, step)
        metrics = self._task_sampler.metrics()
        writer.add_scalar("curriculum/proportion_seen", metrics["proportion_seen"], step)
        writer.add_scalar("curriculum/score", metrics["score"], step)
        for task in self.task_space.tasks:
            writer.add_scalar(f"curriculum/task_{task - 1}_score", metrics["task_scores"][task - 1], step)
            writer.add_scalar(f"curriculum/task_{task - 1}_staleness", metrics["task_staleness"][task - 1], step)
        # task_returns = {task: 0 for task in self.task_space.tasks}
        # for actor, task_return in self._rollouts.returns[]:
        #     writer.add_scalar(f"curriculum/task_{task - 1}_return", self._rollouts.returns[task - 1].mean(), step)
