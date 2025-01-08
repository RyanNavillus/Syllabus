import math
import random
import warnings
from typing import Any, List, Union

import gymnasium as gym
import numpy as np
import torch
from scipy.stats import norm

from syllabus.core import Curriculum
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace


class LearningProgress(Curriculum):
    """
    Provides an interface for tracking success rates of discrete tasks and sampling tasks
    based on their success rate using the method from https://arxiv.org/abs/2106.14876.
    TODO: Support task spaces aside from Discrete
    """

    def __init__(self, *args, eval_envs=None, evaluator=None, ema_alpha=0.1, rnn_shape=None, eval_interval=None, eval_interval_steps=None, eval_eps=1, eval_fn=None, baseline_eval_eps=None, **kwargs):
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

        assert isinstance(
            self.task_space, (DiscreteTaskSpace, MultiDiscreteTaskSpace)
        ), f"LearningProgressCurriculum only supports Discrete and MultiDiscrete task spaces. Got {self.task_space.__class__.__name__}."
        self.random_baseline = None
        self._p_fast = None
        self._p_slow = None
        self._p_true = None
        self.task_rates = None
        self._stale_dist = True

        self.random_baseline = self.eval_and_update(baseline_eval_eps if baseline_eval_eps is not None else eval_eps)

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
        normalized_task_success_rates = np.maximum(
            task_success_rates - self.random_baseline, np.zeros(task_success_rates.shape)) / (1.0 - self.random_baseline)

        if self._p_fast is None:
            # Initial values
            self._p_fast = normalized_task_success_rates
            self._p_slow = normalized_task_success_rates
            self._p_true = task_success_rates
        else:
            # Exponential moving average
            self._p_fast = (normalized_task_success_rates * self.ema_alpha) + (self._p_fast * (1.0 - self.ema_alpha))
            self._p_slow = (self._p_fast * self.ema_alpha) + (self._p_slow * (1.0 - self.ema_alpha))
            self._p_true = (task_success_rates * self.ema_alpha) + (self._p_true * (1.0 - self.ema_alpha))

        self.task_rates = task_success_rates    # Logging only
        self._stale_dist = True
        self.task_dist = None

        return task_success_rates

    def _evaluate_all_tasks(self, eval_episodes=1, verbose=True):
        if verbose:
            print("Evaluating tasks for {eval_episodes} episodes.")
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

                for task, completion, done in zip(task_idx, task_completions, dones):
                    if abs(completion) >= 1.0:
                        task_counts[task] += 1
                        task_successes[task] += math.floor(max(completion, 0.0))
            else:
                warnings.warn("Did not find 'task_completion' in infos. Task success rates will not be evaluated.")

            for done in dones:
                if done:
                    ep_counter += 1
                    if verbose and ep_counter % 100 == 0:
                        print([f"{f}/{g}" for f, g in zip(task_successes, task_counts)])

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

    def _learning_progress(self, reweight: bool = True) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow

        return abs(fast - slow)

    def _reweight(self, p: np.ndarray, p_theta: float = 0.1) -> float:
        """
        Reweight the given success rate using the reweighting function from the paper.
        """
        numerator = p * (1.0 - p_theta)
        denominator = p + p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        """ Sigmoid function for reweighting the learning progress."""
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self) -> List[float]:
        """ Return sampling distribution over the task space based on the learning progress."""
        if not self._stale_dist:
            # No changes since distribution was last computed
            return self.task_dist

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

    def log_metrics(self, writer, logs, step, log_n_tasks=1):
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
            logs.append((f"curriculum/{name}_slow", self._p_slow[idx]))
            logs.append((f"curriculum/{name}_fast", self._p_fast[idx]))
            logs.append((f"curriculum/{name}_lp", learning_progresses[idx]))
        return super().log_metrics(writer, logs, step=step, log_n_tasks=log_n_tasks)


if __name__ == "__main__":
    def sample_binomial(p=0.5, n=200):
        success = 0.0
        for _ in range(n):
            rand = random.random()
            if rand < p:
                success += 1.0
        return success / n

    def generate_history(center=0, curve=1.0, n=100):
        center = center if center else n / 2.0

        def sig(x, x_0=center, curve=curve):
            return 1.0 / (1.0 + math.e**(curve * (x_0 - x)))
        history = []
        probs = []
        success_prob = 0.0
        for i in range(n):
            probs.append(success_prob)
            history.append(sample_binomial(p=success_prob))
            success_prob = sig(i)
        return history, probs

    tasks = range(20)
    histories = {task: generate_history(center=random.randint(0, 100), curve=random.random()) for task in tasks}

    curriculum = LearningProgress(DiscreteTaskSpace(len(tasks)))
    for i in range(len(histories[0][0])):
        for task in tasks:
            curriculum.update_task_progress(task, histories[task][0][i])
        if i > 10:
            distribution = curriculum._sample_distribution()
            print("[", end="")
            for j, prob in enumerate(distribution):
                print(f"{prob:.3f}", end="")
                if j < len(distribution) - 1:
                    print(", ", end="")
            print("]")

    tasks = [0]
    histories = {task: generate_history(n=200, center=75, curve=0.1) for task in tasks}
    curriculum = LearningProgress(DiscreteTaskSpace(len(tasks)))
    lp_raw = []
    lp_reweight = []
    p_fast = []
    p_slow = []
    true_probs = []
    estimates = []
    for estimate, true_prob in zip(histories[0][0], histories[0][1]):
        curriculum.update_task_progress(tasks[0], estimate)
        lp_raw.append(curriculum._learning_progress(tasks[0], reweight=False))
        lp_reweight.append(curriculum._learning_progress(tasks[0]))
        p_fast.append(curriculum._p_fast[0])
        p_slow.append(curriculum._p_slow[0])
        true_probs.append(true_prob)
        estimates.append(estimate)

    try:
        import matplotlib.pyplot as plt

        # TODO: Plot probabilities
        def plot_history(true_probs, estimates, p_slow, p_fast, lp_reweight, lp_raw):
            x_axis = range(0, len(true_probs))
            plt.plot(x_axis, true_probs, color="#222222", label="True Success Probability")
            plt.plot(x_axis, estimates, color="#888888", label="Estimated Success Probability")
            plt.plot(x_axis, p_slow, color="#ee3333", label="p_slow")
            plt.plot(x_axis, p_fast, color="#33ee33", label="p_fast")
            plt.plot(x_axis, lp_raw, color="#c4c25b", label="Learning Progress")
            plt.plot(x_axis, lp_reweight, color="#1544ee", label="Learning Progress Reweighted")
            plt.xlabel('Time step')
            plt.ylabel('Learning Progress')
            plt.legend()
            plt.show()

        plot_history(true_probs, estimates, p_slow, p_fast, lp_reweight, lp_raw)

        # Reweight Plot
        x_axis = np.linspace(0, 1, num=100)
        y_axis = []
        for x in x_axis:
            y_axis.append(curriculum._reweight(x))
        plt.plot(x_axis, y_axis, color="blue", label="p_theta = 0.1")
        plt.xlabel('p')
        plt.ylabel('reweight')
        plt.legend()
        plt.show()

        # Z-score plot
        tasks = [i for i in range(50)]
        curriculum = LearningProgress(DiscreteTaskSpace(len(tasks)))
        histories = {task: generate_history(n=200, center=60, curve=0.09) for task in tasks}
        for i in range(len(histories[0][0])):
            for task in tasks:
                curriculum.update_task_progress(task, histories[task][0][i])
        distribution = curriculum._sample_distribution()
        x_axis = np.linspace(-3, 3, num=len(distribution))
        sigmoid_axis = curriculum._sigmoid(x_axis)
        plt.plot(x_axis, norm.pdf(x_axis, 0, 1), color="blue", label="Normal distribution")
        plt.plot(x_axis, sigmoid_axis, color="orange", label="Sampling weight")
        plt.xlabel('Z-scored distributed learning progress')
        plt.legend()
        plt.show()
    except ImportError:
        warnings.warn("Matplotlib not installed. Plotting will not work.")
