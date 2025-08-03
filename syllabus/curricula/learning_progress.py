import math
import warnings
from itertools import groupby
from typing import Any, List, Optional, Union

import numpy as np

from syllabus.core import Curriculum
from syllabus.core.evaluator import Evaluator
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace, StratifiedDiscreteTaskSpace


def compress_ranges(nums):
    nums = sorted(set(nums))
    ranges = []
    for _, group in groupby(enumerate(nums), lambda x: x[1] - x[0]):
        group = list(group)
        start, end = group[0][1], group[-1][1]
        ranges.append(f"{start}" if start == end else f"{start}-{end}")
    return ", ".join(ranges)


class LearningProgress(Curriculum):
    """
    Provides an interface for tracking success rates of discrete tasks and sampling tasks
    based on their success rate using the method from https://arxiv.org/abs/2106.14876.
    TODO: Support task spaces aside from Discrete
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *args,
        ema_alpha: float = 0.1,
        p_theta: float = 0.1,
        eval_interval: Optional[int] = None,
        eval_interval_steps: Optional[int] = None,
        eval_eps: float = 1,
        baseline_eval_eps: Optional[float] = None,
        normalize_success: bool = True,
        continuous_progress: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.evaluator = evaluator
        self.ema_alpha = ema_alpha
        self.p_theta = p_theta
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
        self._p_fast = None
        self._p_slow = None
        self._p_true = None
        self.task_rates = None
        self.task_dist = None
        self._stale_dist = True
        self._baseline_eval_eps = baseline_eval_eps if baseline_eval_eps is not None else eval_eps

    def eval_and_update(self, eval_eps=1):
        _, task_success_rates, final_success_rates, _ = self.evaluator.evaluate_agent(eval_eps, verbose=True)
        task_success_rates = final_success_rates.numpy() if self.continuous_progress else task_success_rates.numpy()

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

        task_rates = self.normalized_task_success_rates if self.normalize_success else task_success_rates

        if self._p_fast is None:
            # Initial values
            self._p_fast = task_rates
            self._p_slow = task_rates
            self._p_true = task_success_rates
        else:
            # Exponential moving average
            self._p_fast = (task_rates * self.ema_alpha) + (self._p_fast * (1.0 - self.ema_alpha))
            self._p_slow = (self._p_fast * self.ema_alpha) + (self._p_slow * (1.0 - self.ema_alpha))
            self._p_true = (task_success_rates * self.ema_alpha) + (self._p_true * (1.0 - self.ema_alpha))

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

    def _learning_progress(self, reweight: bool = True) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow

        return abs(fast - slow)

    def _reweight(self, p: np.ndarray) -> float:
        """
        Reweight the given success rate using the reweighting function from the paper.
        """
        numerator = p * (1.0 - self.p_theta)
        denominator = p + self.p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        """ Sigmoid function for reweighting the learning progress."""
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self) -> List[float]:
        """ Return sampling distribution over the task space based on the learning progress."""
        if not self._stale_dist:
            # No changes since distribution was last computed
            return self.task_dist

        if self.task_rates is None:
            self.eval_and_update(self._baseline_eval_eps)

        task_dist = np.ones(self.num_tasks) / self.num_tasks

        learning_progress = self._learning_progress()
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(posidxs) > 0

        subprobs = learning_progress[posidxs] if any_progress else learning_progress
        std = np.std(subprobs)
        subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1)  # z-score
        subprobs = self._sigmoid(subprobs)  # sigmoid
        subprobs = subprobs / np.sum(subprobs)  # normalize
        if any_progress:
            # If some tasks have nonzero progress, zero out the rest
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            # If all tasks have 0 progress, return uniform distribution
            task_dist = subprobs

        self.task_dist = task_dist
        self._stale_dist = False
        return task_dist

    def log_metrics(self, writer, logs, step, log_n_tasks=-1):
        logs = [] if logs is None else logs
        learning_progresses = self._learning_progress()
        logs.append(("curriculum/learning_progress", np.mean(learning_progresses)))
        if self.task_rates is not None:
            logs.append(("curriculum/mean_success_rate", np.mean(self.task_rates)))

        tasks = range(self.num_tasks)
        if self.num_tasks > log_n_tasks and log_n_tasks != -1:
            warnings.warn(f"Too many tasks to log {self.num_tasks}. Only logging stats for 1 task.", stacklevel=2)
            tasks = tasks[:log_n_tasks]

        for idx in tasks:
            name = self.task_names(self.tasks[idx], idx)
            logs.append((f"curriculum/{name}_success_rate", self.task_rates[idx]))
            logs.append((f"curriculum/{name}_lp", learning_progresses[idx]))
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)


class StratifiedLearningProgress(LearningProgress):
    def __init__(self, *args, selection_metric="success", **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.task_space, StratifiedDiscreteTaskSpace)
        assert selection_metric in [
            "success", "score", "learnability"], f"Selection metric {selection_metric} not recognized. Use 'success', 'score', or 'learnability'."
        self.selection_metric = selection_metric

    def _sample_distribution(self) -> List[float]:
        # Prioritize tasks by learning progress first
        lp_dist = super()._sample_distribution()
        selection_weight = np.ones(len(lp_dist)) * 0.001
        if self.selection_metric == "learnability":
            metric = self.task_rates * (1.0 - self.task_rates)
        elif self.selection_metric == "score":
            metric = lp_dist
        else:
            metric = self.task_rates

        # Find the highest success rate task in each strata
        for strata in self.task_space.strata:
            task_idx = np.argsort(metric[np.array(list(strata))])[-1]
            selection_weight[strata[task_idx]] = 1.0

        # Scale and normalize
        stratified_dist = lp_dist * selection_weight
        stratified_dist = stratified_dist / np.sum(stratified_dist)
        return stratified_dist


class StratifiedDomainRandomization(LearningProgress):
    def __init__(self, *args, selection_metric="success", **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.task_space, StratifiedDiscreteTaskSpace)
        assert selection_metric in ["success", "progress"]
        self.selection_metric = selection_metric

    def _sample_distribution(self) -> List[float]:
        if not self._stale_dist:
            # No changes since distribution was last computed
            return self.task_dist

        if self.task_rates is None:
            self.eval_and_update(self._baseline_eval_eps)

        # Prioritize tasks uniformly first
        uni_dist = np.ones(self.num_tasks) / self.num_tasks
        selection_weight = np.ones(len(uni_dist)) * 0.0001
        metric = self.task_rates if self.selection_metric == "success" else uni_dist

        # Find the highest success rate task in each strata
        for strata in self.task_space.strata:
            task_idx = np.argsort(metric[np.array(list(strata))])[-1]
            selection_weight[strata[task_idx]] = 1.0

        # Scale and normalize
        stratified_dist = uni_dist * selection_weight
        stratified_dist = stratified_dist / np.sum(stratified_dist)

        self.task_dist = stratified_dist
        self._stale_dist = False
        return stratified_dist
