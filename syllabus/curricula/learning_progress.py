import math
import random
import warnings
from typing import List

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from scipy.stats import norm

from syllabus.core import Curriculum
from syllabus.task_space import TaskSpace


class LearningProgressCurriculum(Curriculum):
    """
    Provides an interface for tracking success rates of discrete tasks and sampling tasks
    based on their success rate using the method from https://arxiv.org/abs/2106.14876.
    TODO: Support task spaces aside from Discrete
    """
    REQUIRES_STEP_UPDATES = False
    REQUIRES_EPISODE_UPDATES = False
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, *args, ema_alpha=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_alpha = ema_alpha

        assert isinstance(self.task_space.gym_space, (Discrete, MultiDiscrete))
        self._p_fast = np.zeros(self.num_tasks)
        self._p_slow = np.zeros(self.num_tasks)

    def update_task_progress(self, task: int, progress: float, env_id: int = None):
        """
        Update the success rate for the given task using a fast and slow exponential moving average.
        """
        if task is None or progress == 0.0:
            return
        super().update_task_progress(task, progress)

        self._p_fast[task] = (progress * self.ema_alpha) + (self._p_fast[task] * (1.0 - self.ema_alpha))
        self._p_slow[task] = (self._p_fast[task] * self.ema_alpha) + (self._p_slow[task] * (1.0 - self.ema_alpha))

    def _learning_progress(self, task: int, reweight: bool = True) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        slow = self._reweight(self._p_slow[task]) if reweight else self._p_slow[task]
        fast = self._reweight(self._p_fast[task]) if reweight else self._p_fast[task]
        return abs(fast - slow)

    def _reweight(self, p: np.ndarray, p_theta: float = 0.1) -> float:
        """
        Reweight the given success rate using the reweighting function from the paper.
        """
        numerator = p * (1.0 - p_theta)
        denominator = p + p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self) -> List[float]:
        if self.num_tasks == 0:
            return []

        task_dist = np.ones(self.num_tasks) / self.num_tasks

        task_lps = self._learning_progress(np.asarray(self.tasks))
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

    def on_step(self, obs, rew, term, trunc, info) -> None:
        """
        Update the curriculum with the current step results from the environment.
        """
        pass


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

    curriculum = LearningProgressCurriculum(TaskSpace(len(tasks)))
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
    curriculum = LearningProgressCurriculum(TaskSpace(len(tasks)))
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
        curriculum = LearningProgressCurriculum(TaskSpace(len(tasks)))
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
