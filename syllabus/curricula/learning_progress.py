import math
import random
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
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
    REQUIRES_CENTRAL_UPDATES = False

    def __init__(self, task_space: TaskSpace, **kwargs):
        super().__init__(task_space, **kwargs)
        # Save task list to file to coordinate between multiple processes
        self._p_slow = defaultdict(float)    # Map from task to slow EMA process
        self._p_fast = defaultdict(float)    # Map from task to fast EMA process
        self.task_space = task_space
        if isinstance(self.task_space.gym_space, Discrete) or isinstance(self.task_space.gym_space, MultiDiscrete):
            for task in self.tasks:
                self._p_fast[task] = 0.0
                self._p_slow[task] = 0.0
        #print(f"Creating curriculum with {self._n_tasks} tasks: {self._tasks} from task space: {self.task_space}")

    def update_task_progress(self, task: int, progress: float):
        """
        Update the success rate for the given task using a fast and slow exponential moving average.
        """
        if task is None:
            return
        super().update_task_progress(task, progress)

        k_slow = 2.0 / (55 + 1.0)
        k_fast = 2.0 / (30 + 1.0)
        self._p_slow[task] = (progress * k_slow) + (self._p_slow[task] * (1.0 - k_slow))
        self._p_fast[task] = (progress * k_fast) + (self._p_fast[task] * (1.0 - k_fast))

    def _lp_metric(self, task: int, reweight: bool = True) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        slow = self._reweight(self._p_slow[task]) if reweight else self._p_slow[task]
        fast = self._reweight(self._p_fast[task]) if reweight else self._p_fast[task]
        return abs(fast - slow)

    def _reweight(self, p: float, p_theta: float = 0.1) -> float:
        """
        Reweight the given success rate using the reweighting function from the paper.
        """
        numerator = float(p) * (1.0 - p_theta)
        denominator = float(p) + (p_theta * (1.0 - (2.0 * float(p))))
        return numerator / denominator

    def _sigmoid(self, X: List[float], center: float = 0.0, curve: float = 1.0) -> List[float]:
        def sig(x, x_0=center, curve=curve):
            return 1.0 / (1.0 + math.e**(curve * (x_0 - x)))
        return [sig(x) for x in X]

    def _softmax(self, X: List[float]) -> List[float]:
        exp_sum = sum([x * math.e**x for x in X])
        return [(x * math.e**x / exp_sum) for x in X]

    # @property
    # def _n_tasks(self):
    #     return len(self._p_slow.keys())

    # @property
    # def _tasks(self):
    #     return self._p_slow.keys()

    def _sample_distribution(self) -> List[float]:
        if self.num_tasks == 0:
            return []

        task_lps = []
        for task in self.tasks:
            task_lps.append(self._lp_metric(task))

        # Standardize
        task_lps_mean = sum(task_lps) / len(task_lps)
        task_lps_sqr = [task_lp**2 for task_lp in task_lps]
        task_lps_std = math.sqrt(sum(task_lps_sqr))

        # If tasks all have the same lp, return uniform distribution
        if task_lps_std == 0:
            return [1 / self.num_tasks for _ in self.tasks]
        task_lps_standard = [(task_lp - task_lps_mean) / task_lps_std for task_lp in task_lps]

        # Sigmoid
        task_lps_sigmoid = self._sigmoid(task_lps_standard, center=1.28, curve=3.0)

        # Softmax - convert weights to sampling probabilities
        task_dist = self._softmax(task_lps_sigmoid)
        return task_dist

    def on_step(self, obs, rew, done, info) -> None:
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

    tasks = range(20)
    histories = {task: generate_history(center=random.randint(0, 100), curve=random.random()) for task in tasks}

    curriculum = LearningProgressCurriculum(TaskSpace(len(tasks)))
    for i in range(len(histories[0][0])):
        for task in tasks:
            curriculum.update_task_progress(task, histories[task][0][i])
        if i > 10:
            distribution, _ = curriculum._sample_distribution()
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
        lp_raw.append(curriculum._lp_metric(tasks[0], reweight=False))
        lp_reweight.append(curriculum._lp_metric(tasks[0]))
        p_fast.append(curriculum._p_fast[0])
        p_slow.append(curriculum._p_slow[0])
        true_probs.append(true_prob)
        estimates.append(estimate)

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
    distribution, task_lps_standardized = curriculum._sample_distribution()
    x_axis = np.linspace(-3, 3, num=len(task_lps_standardized))
    sigmoid_axis = curriculum._sigmoid(x_axis, center=1.28, curve=3.0)
    plt.plot(x_axis, norm.pdf(x_axis, 0, 1), color="blue", label="Normal distr")
    plt.plot(x_axis, sigmoid_axis, color="orange", label="Sampling weight")
    plt.xlabel('Z-scored distributed learning progress')
    plt.legend()
    plt.show()
