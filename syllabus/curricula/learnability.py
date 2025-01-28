import math
import random
import warnings
from typing import Any, List, Union

import gymnasium as gym
import numpy as np
import torch
from scipy.stats import norm

from syllabus.core import Curriculum
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace, StratifiedDiscreteTaskSpace
from syllabus.utils import UsageError


class Learnability(Curriculum):
    """
    Provides an interface for tracking success rates of discrete tasks and sampling tasks
    based on their success rate using the method from https://arxiv.org/abs/2106.14876.
    TODO: Support task spaces aside from Discrete
    """

    def __init__(self, *args, eval_envs=None, evaluator=None, ema_alpha=0.1, rnn_shape=None, eval_interval=None, eval_interval_steps=None, eval_eps=1, eval_fn=None, baseline_eval_eps=None, normalize_success=False, continuous_progress=False, buffer_size=1000, learnable_prob=1.0, sampling="topk", **kwargs):
        super().__init__(*args, **kwargs)
        assert (eval_envs is not None and evaluator is not None) or eval_fn is not None, "Either eval_envs and evaluator or eval_fn must be provided."
        # Decide evaluation method
        if eval_fn is None:
            self.custom_eval = False
            self.eval_envs = eval_envs
            self.evaluator = evaluator
            self._evaluate = self._evaluate_all_tasks
        else:
            self.custom_eval = True
            self._evaluate = eval_fn

        self.ema_alpha = ema_alpha
        self.lstm_shape = rnn_shape
        self.eval_interval = eval_interval
        assert eval_interval is None or eval_interval_steps is None, "Only one of eval_interval or eval_interval_steps can be set."
        self.eval_interval_steps = eval_interval_steps
        self.eval_eps = eval_eps
        self.completed_episodes = 0
        self.current_steps = 0
        self.normalize_success = normalize_success
        self.buffer_size = buffer_size
        self.learnable_prob = learnable_prob
        self.continuous_progress = continuous_progress
        self.sampling = sampling

        assert isinstance(
            self.task_space, (DiscreteTaskSpace, MultiDiscreteTaskSpace)
        ), f"LearningProgressCurriculum only supports Discrete and MultiDiscrete task spaces. Got {self.task_space.__class__.__name__}."
        self.random_baseline = None
        self.task_rates = None
        self.task_dist = None
        self._stale_dist = True

        self.eval_and_update(baseline_eval_eps if baseline_eval_eps is not None else eval_eps)

    def eval_and_update(self, eval_eps=1):
        task_success_rates = self._evaluate(eval_episodes=int(eval_eps))

        if self.random_baseline is None:
            # Assume that any perfect success rate is actually 75% due to evaluation precision.
            # Prevents NaN probabilities and prevents task from being completely ignored.
            high_success_idxs = np.where(task_success_rates > 0.75)
            high_success_rates = task_success_rates[high_success_idxs]
            warnings.warn(
                f"Tasks {high_success_idxs} had very high success rates {high_success_rates} for random baseline. Consider removing them from the training set of tasks.")
            self.random_baseline = np.minimum(task_success_rates, 0.75)

        # Update task scores
        self.normalized_task_success_rates = np.maximum(
            task_success_rates - self.random_baseline, np.zeros(task_success_rates.shape)) / (1.0 - self.random_baseline)

        self.task_rates = task_success_rates    # Used for logging and OMNI
        self._stale_dist = True
        self.task_dist = None

        return task_success_rates

    def _evaluate_all_tasks(self, eval_episodes=1, verbose=True):
        if verbose:
            print(f"Evaluating tasks for {eval_episodes} episodes.")
        task_success_rates = np.zeros(self.task_space.num_tasks)
        lstm_state = torch.zeros(*self.lstm_shape) if self.lstm_shape else None
        obss, _ = self.eval_envs.reset()
        ep_counter = 0
        dones = [False] * self.eval_envs.num_envs
        task_counts = np.zeros(self.task_space.num_tasks)   # TODO: Maybe start with assigned tasks?
        task_successes = np.zeros(self.task_space.num_tasks)

        while ep_counter < eval_episodes:
            actions, lstm_state, _ = self.evaluator.get_action(obss, lstm_state=lstm_state, done=dones)
            actions = torch.flatten(actions)
            if isinstance(self.eval_envs.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
                actions = actions.int()
            obss, rewards, terminateds, truncateds, infos = self.eval_envs.step(actions.cpu().numpy())
            dones = tuple(a | b for a, b in zip(terminateds, truncateds))

            if "task_completion" in infos:
                task_completions = infos["task_completion"]
                task_idx = infos["task"]
                # print(task_completions)

                for task, completion, done in zip(task_idx, task_completions, dones):
                    # Completion < 0 is considered a failure
                    if self.continuous_progress:
                        # Continuous progress can only be measured at the end of the episode
                        if done:
                            task_counts[task] += 1
                            task_successes[task] += max(completion, 0.0)
                    else:
                        # Binary success/failure can be measured at each step.
                        # Assumes that success will only be 1.0 or -1.0 for 1 step
                        if abs(completion) >= 1.0:
                            task_counts[task] += 1
                            task_successes[task] += math.floor(max(completion, 0.0))
            else:
                raise UsageError("Did not find 'task_completion' in infos. Task success rates will not be evaluated.")

            for done in dones:
                if done:
                    ep_counter += 1
                    if verbose and ep_counter % 100 == 0:
                        print([f"{f:.1f}/{g:.0f}" for f, g in zip(task_successes, task_counts)])

        # Warn user if any task_counts are 0
        if np.any(task_counts == 0):
            warnings.warn(
                f"Tasks {np.where(task_counts == 0)} were not attempted during evaluation. Consider increasing eval episodes.")

        task_counts = np.maximum(task_counts, np.ones_like(task_counts))
        task_success_rates = np.divide(task_successes, task_counts)
        return task_success_rates

    def update_task_progress(self, task: int, progress: Union[float, bool], env_id: int = None):
        """
        Update the success rate for the given task using a fast and slow exponential moving average.
        """
        super().update_task_progress(task, progress)

    def update_on_episode(self, episode_return: float, length: int, task: Any, progress: Union[float, bool], env_id: int = None) -> None:
        self.completed_episodes += 1
        self.current_steps += length
        if self.eval_interval is not None and self.completed_episodes % self.eval_interval == 0:
            self.eval_and_update(eval_eps=self.eval_eps)
        if self.eval_interval_steps is not None and self.current_steps > self.eval_interval_steps:
            self.eval_and_update(eval_eps=self.eval_eps)
            self.current_steps = 0

    def _learnability(self) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        success_rates = self.normalized_task_success_rates if self.normalize_success else self.task_rates
        return success_rates * (1 - success_rates)

    def _sample_distribution(self) -> List[float]:
        """ Return sampling distribution over the task space based on the learning progress."""
        if not self._stale_dist:
            # No changes since distribution was last computed
            return self.task_dist

        learnability = self._learnability()
        if self.sampling == "topk":
            highest_learnability_tasks = np.argsort(learnability)[::-1]
            top_k = highest_learnability_tasks[:self.buffer_size]

            learnable_task_dist = np.zeros(self.num_tasks)
            learnable_task_dist[top_k] = 1.0 / self.buffer_size

            uniform_task_dist = np.ones(self.num_tasks) / self.num_tasks
            task_dist = self.learnable_prob * learnable_task_dist + (1 - self.learnable_prob) * uniform_task_dist
        elif self.sampling == "dist":
            task_dist = learnability if np.sum(learnability) > 0 else np.ones(self.num_tasks)

        task_dist = task_dist / np.sum(task_dist)
        self.task_dist = task_dist
        self._stale_dist = False
        return task_dist

    def log_metrics(self, writer, logs, step, log_n_tasks=1):
        logs = [] if logs is None else logs
        learnability = self._learnability()
        logs.append(("curriculum/learning_progress", np.mean(learnability)))
        if self.task_rates is not None:
            logs.append(("curriculum/mean_success_rate", np.mean(self.task_rates)))

        tasks = range(self.num_tasks)
        if self.num_tasks > log_n_tasks and log_n_tasks != -1:
            warnings.warn(f"Too many tasks to log {self.num_tasks}. Only logging stats for 1 task.", stacklevel=2)
            tasks = tasks[:log_n_tasks]

        for idx in tasks:
            name = self.task_names(self.tasks[idx], idx)
            logs.append((f"curriculum/{name}_lp", learnability[idx]))
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)


class StratifiedLearnability(Learnability):
    def __init__(self, *args, selection_metric="success", **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.task_space, StratifiedDiscreteTaskSpace)
        assert selection_metric in ["success", "progress"]
        self.selection_metric = selection_metric

    def _sample_distribution(self) -> List[float]:
        # Prioritize tasks by learning progress first
        lp_dist = super()._sample_distribution()
        selection_weight = np.ones(len(lp_dist)) * 0.0001
        metric = self.task_rates if self.selection_metric == "success" else lp_dist

        # Find the highest success rate task in each strata
        for strata in self.task_space.strata:
            task_idx = np.argsort(metric[np.array(list(strata))])[-1]
            selection_weight[strata[task_idx]] = 1.0
        # Scale and normalize
        stratified_dist = lp_dist * selection_weight
        stratified_dist = stratified_dist / np.sum(stratified_dist)
        return stratified_dist
