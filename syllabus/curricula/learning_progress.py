import math
import random
import warnings
from typing import Any, List, Union

import numpy as np
from scipy.stats import norm

from syllabus.core import Curriculum
from syllabus.task_space import DiscreteTaskSpace, MultiDiscreteTaskSpace


class LearningProgress(Curriculum):
    """
    Provides an interface for tracking success rates of discrete tasks and sampling tasks 
    based on their success rate using the method from https://arxiv.org/abs/2106.14876.
    TODO: Support task spaces aside from Discrete
    """

    def __init__(self, eval_envs, evaluator, *args, ema_alpha=0.1, eval_interval=None, eval_interval_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_envs = eval_envs
        self.evaluator = evaluator
        self.ema_alpha = ema_alpha
        self.eval_interval = eval_interval
        self.eval_interval_steps = eval_interval_steps
        assert self.eval_interval is None or self.eval_interval_steps is None, "Only one of eval_interval or eval_interval_steps can be set."
        self.completed_episodes = 0
        self.completed_steps = 0

        assert isinstance(
            self.task_space, (DiscreteTaskSpace, MultiDiscreteTaskSpace)
        ), f"LearningProgressCurriculum only supports Discrete and MultiDiscrete task spaces. Got {self.task_space.__class__.__name__}."
        self._p_fast = np.zeros(self.num_tasks)
        self._p_slow = np.zeros(self.num_tasks)

        self._evaluate_all_tasks()

    def _evaluate_all_tasks(self, eval_eps=1):
        task_progresses = np.zeros(self.task_space.num_tasks)
        for task_idx, task in enumerate(self.task_space.tasks):
            obss, _ = self.eval_envs.reset(options=task)
            ep_counter = 0
            progress = 0.0
            while ep_counter < eval_eps:
                actions, _, _ = self.evaluator.get_action(obss)
                obss, rewards, terminateds, truncateds, infos = self.eval_envs.step(actions)
                dones = tuple(a | b for a, b in zip(terminateds, truncateds))
                for i, done in enumerate(dones):
                    if done:
                        if isinstance(infos, list):
                            task_progress = infos[i]["final_info"]['task_completion']
                        elif isinstance(infos, dict):
                            task_progress = infos["final_info"][i]['task_completion']
                        progress += task_progress
                        ep_counter += 1
            task_progresses[task_idx] = progress
        task_success_rates = np.divide(task_progresses, float(eval_eps))

        # Update task scores
        self._p_fast = (task_progresses * self.ema_alpha) + (self._p_fast * (1.0 - self.ema_alpha))
        self._p_slow = (self._p_fast * self.ema_alpha) + (self._p_slow * (1.0 - self.ema_alpha))

        return task_success_rates

    def update_task_progress(self, task: int, progress: Union[float, bool], env_id: int = None):
        """
        Update the success rate for the given task using a fast and slow exponential moving average.
        """
        if task is None or progress == 0.0:
            return
        super().update_task_progress(task, progress)

        self._p_fast[task] = (progress * self.ema_alpha) + (self._p_fast[task] * (1.0 - self.ema_alpha))
        self._p_slow[task] = (self._p_fast[task] * self.ema_alpha) + (self._p_slow[task] * (1.0 - self.ema_alpha))

    def update_on_episode(self, episode_return: float, length: int, task: Any, progress: Union[float, bool], env_id: int = None) -> None:
        self.completed_episodes += 1
        self.completed_steps += length
        if self.eval_interval is not None and self.completed_episodes % self.eval_interval == 0:
            self._evaluate_all_tasks()
        if self.eval_interval_steps is not None and self.completed_steps > self.eval_interval_steps:
            self._evaluate_all_tasks()
            self.completed_steps = 0

    def _learning_progress(self, reweight: bool = True) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        slow = self._reweight(self._p_slow) if reweight else self._p_slow
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
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
        if self.num_tasks == 0:
            return []

        task_dist = np.ones(self.num_tasks) / self.num_tasks

        task_lps = self._learning_progress()
        posidxs = [i for i, lp in enumerate(task_lps) if lp > 0]
        zeroout = len(posidxs) > 0

        subprobs = task_lps[posidxs] if zeroout else task_lps
        std = np.std(subprobs)
        subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1)  # z-score
        subprobs = self._sigmoid(subprobs)  # sigmoid
        subprobs = subprobs / np.sum(subprobs)  # normalize
        if zeroout:
            # If some tasks have nonzero progress, zero out the rest
            task_dist = np.zeros(len(task_lps))
            task_dist[posidxs] = subprobs
        else:
            # If all tasks have 0 progress, return uniform distribution
            task_dist = subprobs
        return task_dist


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
