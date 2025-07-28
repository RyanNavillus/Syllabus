import math
import warnings
from typing import Any, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from syllabus.core import Curriculum
from syllabus.core.evaluator import Evaluator
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace, StratifiedDiscreteTaskSpace
from syllabus.utils import UsageError


class Learnability(Curriculum):
    """
    Provides an interface for tracking success rates of discrete tasks and sampling tasks
    based on their success rate using the method from https://arxiv.org/abs/2106.14876.
    TODO: Support task spaces aside from Discrete
    """

    def __init__(
        self,
            evaluator: Evaluator,
            *args,
            topk: int = 1000,
            learnable_prob: float = 1.0,
            sampling: str = "topk",
            eval_interval: Optional[int] = None,
            eval_interval_steps: Optional[int] = None,
            eval_eps: float = 1,
            baseline_eval_eps: Optional[float] = None,
            normalize_success: bool = False,
            continuous_progress: bool = False,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.evaluator = evaluator
        self.k = topk
        self.learnable_prob = learnable_prob
        self.sampling = sampling
        self.eval_interval = eval_interval
        assert eval_interval is None or eval_interval_steps is None, "Only one of eval_interval or eval_interval_steps can be set."
        self.eval_interval_steps = eval_interval_steps
        self.eval_eps = eval_eps
        self.completed_episodes = 0
        self.current_steps = 0
        self.normalize_success = normalize_success
        self.continuous_progress = continuous_progress
        self.normalized_task_success_rates = None

        assert isinstance(
            self.task_space, (DiscreteTaskSpace, MultiDiscreteTaskSpace)
        ), f"LearningProgressCurriculum only supports Discrete and MultiDiscrete task spaces. Got {self.task_space.__class__.__name__}."
        self.random_baseline = None
        self.task_rates = None
        self.task_dist = None
        self._stale_dist = True
        self._baseline_eval_eps = baseline_eval_eps if baseline_eval_eps is not None else eval_eps

    def eval_and_update(self, eval_eps=1):
        _, task_success_rates, final_success_rates, _ = self.evaluator.evaluate_agent(eval_eps, verbose=True)
        if self.continuous_progress:
            task_success_rates = final_success_rates

        if self.random_baseline is None:
            # Assume that any perfect success rate is actually 75% due to evaluation precision.
            # Prevents NaN probabilities and prevents task from being completely ignored.
            high_success_idxs = np.where(task_success_rates > 0.75)[0]
            high_success_rates = task_success_rates[high_success_idxs]
            if len(high_success_idxs) > 0:
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

        if self.task_rates is None:
            self.eval_and_update(self._baseline_eval_eps)

        learnability = self._learnability()
        if self.sampling == "topk":
            highest_learnability_tasks = np.argsort(learnability)[::-1]
            top_k = highest_learnability_tasks[:self.k]

            learnable_task_dist = np.zeros(self.num_tasks)
            learnable_task_dist[top_k] = 1.0 / self.k

            uniform_task_dist = np.ones(self.num_tasks) / self.num_tasks
            task_dist = self.learnable_prob * learnable_task_dist + (1 - self.learnable_prob) * uniform_task_dist
        elif self.sampling == "dist":
            task_dist = learnability if np.sum(learnability) > 0 else np.ones(self.num_tasks)
        else:
            raise UsageError(f"Sampling method {self.sampling} not recognized. Use 'topk' or 'dist'.")

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
            logs.append((f"curriculum/{name}_success_rate", self.task_rates[idx]))
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
