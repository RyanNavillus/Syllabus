import warnings
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np

from syllabus.task_space import TaskSpace

from .stat_recorder import StatRecorder

Agent = TypeVar("Agent")


class Curriculum:
    """Base class and API for defining curricula to interface with Gym environments.
    """

    def __init__(self, task_space: TaskSpace, random_start_tasks: int = 0, task_names: Callable = None, record_stats: bool = False, pie_plot_interval: int = 1) -> None:
        """Initialize the base Curriculum

        :param task_space: the environment's task space from which new tasks are sampled
        :param random_start_tasks: Number of uniform random tasks to sample before using the algorithm's sample method, defaults to 0
        :param task_names: Names of the tasks in the task space, defaults to None
        :param record_stats: Whether to record statistics for each task, defaults to False
        """
        assert isinstance(
            task_space, TaskSpace), f"task_space must be a TaskSpace object. Got {type(task_space)} instead."
        self.task_space = task_space
        self.random_start_tasks = random_start_tasks
        self.completed_tasks = 0
        self.task_names = task_names if task_names is not None else lambda task, idx: idx
        self.stat_recorder = StatRecorder(self.task_space, task_names=task_names) if record_stats else None
        self.pie_plot_interval = pie_plot_interval
        self.pie_plot_counter = 0

        if self.num_tasks == 0:
            warnings.warn("Task space is empty. This will cause errors during sampling if no tasks are added.", stacklevel=2)

        # Log wandb table
        try:
            import wandb
            self.wandb_table = wandb.Table(columns=["global_step", "Top tasks", "Probabilities"], log_mode='MUTABLE')
        except ImportError:
            pass

    @property
    def requires_step_updates(self) -> bool:
        """Returns whether the curriculum requires step updates from the environment.

        :return: True if the curriculum requires step updates, False otherwise
        """
        return False

    @property
    def num_tasks(self) -> int:
        """Counts the number of tasks in the task space.

        :return: Returns the number of tasks in the task space if it is countable, TODO: -1 otherwise
        """
        return self.task_space.num_tasks

    @property
    def tasks(self) -> List[tuple]:
        """List all of the tasks in the task space.

        :return: List of tasks if task space is enumerable, TODO: empty list otherwise?
        """
        return self.task_space.tasks

    def update_task_progress(self, task: Any, progress: Union[float, bool], env_id: int = None) -> None:
        """Update the curriculum with a task and its progress. This is used for binary tasks that can be completed mid-episode.

        :param task: Task for which progress is being updated.
        :param progress: Progress toward completion or success rate of the given task. 1.0 or True typically indicates a complete task.
        :param env_id: Environment identifier
        """
        self.completed_tasks += 1

    def update_on_step(self, task: Any, obs: Any, rew: float, term: bool, trunc: bool, info: dict, progress: Union[float, bool], env_id: int = None) -> None:
        """ Update the curriculum with the current step results from the environment.

        :param obs: Observation from the environment
        :param rew: Reward from the environment
        :param term: True if the episode ended on this step, False otherwise
        :param trunc: True if the episode was truncated on this step, False otherwise
        :param info: Extra information from the environment
        :param progress: Progress toward completion or success rate of the given task. 1.0 or True typically indicates a complete task.
        :param env_id: Environment identifier
        :raises NotImplementedError:
        """
        raise NotImplementedError(
            "This curriculum does not require step updates. Set update_on_step for the environment sync wrapper to False to improve performance and prevent this error.")

    def update_on_step_batch(self, step_results: Tuple[List[Any], List[Any], List[int], List[bool], List[bool], List[Dict], List[int]], env_id: int = None) -> None:
        """Update the curriculum with a batch of step results from the environment.

        This method can be overridden to provide a more efficient implementation. It is used
        as a convenience function and to optimize the multiprocessing message passing throughput.

        :param step_results: List of step results
        :param env_id: Environment identifier
        """
        tasks, obs, rews, terms, truncs, infos, progresses = tuple(step_results)
        for t, o, r, te, tr, i, p in zip(tasks, obs, rews, terms, truncs, infos, progresses):
            self.update_on_step(t, o, r, te, tr, i, p, env_id=env_id)

    def update_on_episode(self, episode_return: float, length: int, task: Any, progress: Union[float, bool], env_id: int = None) -> None:
        """Update the curriculum with episode results from the environment.

        :param episode_return: Episodic return
        :param length: Length of the episode
        :param task: Task for which the episode was completed
        :param progress: Progress toward completion or success rate of the given task. 1.0 or True typically indicates a complete task.
        :param env_id: Environment identifier
        """
        if self.stat_recorder is not None:
            self.stat_recorder.record(episode_return, length, task, env_id)

    def get_agent(self, agent_id: int) -> Agent:
        """ Load an agent from the buffer of saved agents.

        :param agent_id: Identifier of the agent to load
        :return: Loaded agent
        """
        raise NotImplementedError("This curriculum does not track agents.")

    def add_agent(self, agent: Agent):
        """ Add an agent to the curriculum.

        :param agent: Agent to add to the curriculum
        :return agent_id: Identifier of the added agent
        """
        raise NotImplementedError("This curriculum does not track agents.")

    def _sample_distribution(self) -> List[float]:
        """Returns a sample distribution over the task space.

        Any curriculum that maintains a true probability distribution should implement this method to retrieve it.
        """
        raise NotImplementedError

    def _should_use_startup_sampling(self) -> bool:
        return self.random_start_tasks > 0 and self.completed_tasks < self.random_start_tasks

    def _startup_sample(self, k: int = 1) -> List[Any]:
        return [self.task_space.encode(self.task_space.sample()) for _ in range(k)]

    def sample(self, k: int = 1) -> List[Any]:
        """Sample k tasks from the curriculum.

        :param k: Number of tasks to sample, defaults to 1
        :return: Either returns a single task if k=1, or a list of k tasks
        """

        if self._should_use_startup_sampling():
            return self._startup_sample()

        # Use list of indices because np.choice does not play nice with tuple tasks
        task_dist = self._sample_distribution()
        task_idx = np.random.choice(list(range(self.num_tasks)), size=k, p=task_dist)
        return task_idx

    def normalize(self, reward: float, task: Any) -> float:
        """
        Normalize reward by task.

        :param reward: Reward to normalize
        :param task: Task for which the reward was received
        :return: Normalized reward
        """
        assert self.stat_recorder is not None, "Curriculum must be initialized with record_stats=True to use normalize()"
        return self.stat_recorder.normalize(reward, task)

    def plot_pie_chart(self, task_dist, run_id, log_n_tasks: int = 1):
        # Identify tasks above the 1% threshold
        threshold = 0.01
        eligible_indices = [i for i, p in enumerate(task_dist) if p > threshold]

        # Sort eligible tasks by probability, descending
        sorted_idx = sorted(
            eligible_indices,
            key=lambda i: task_dist[i],
            reverse=True
        )

        top_indices = sorted_idx[:log_n_tasks] if log_n_tasks != -1 else sorted_idx

        # Gather the top-n probs and corresponding labels
        top_probs = [task_dist[i] for i in top_indices]
        top_labels = [self.task_names(self.tasks[i], i) for i in top_indices]

        # Compute other probability
        other_prob = max(0.0, 1.0 - sum(top_probs))
        if other_prob > 0:
            top_probs.append(other_prob)
            top_labels.append("Other")

        # Generate task distribution pie chart
        fig, ax = plt.subplots()
        ax.pie(top_probs, labels=top_labels, autopct="%1.1f%%", startangle=0)
        ax.set_title("Task Sampling Distribution")
        plt.savefig(f'tmp_syllabus_fig_{run_id}.png')
        return fig

    def log_metrics(self, writer, logs: List[Dict], step: int = None, log_n_tasks: int = -1):
        """Log the task distribution to the provided writer.

        :param writer: Tensorboard summary writer or wandb object
        :param logs: Cumulative list of logs to write
        :param step: Global step number
        :param log_n_tasks: Maximum number of tasks to log, defaults to 1. Use -1 to log all tasks.
        :return: Updated logs list
        """
        logs = [] if logs is None else logs
        if self.stat_recorder is not None:
            logs += self.stat_recorder.get_metrics(log_n_tasks=log_n_tasks)

        try:
            import wandb
            use_wandb = writer == wandb
        except ImportError:
            use_wandb = False

        try:
            # Use uniform distribution during startup sampling
            task_dist = self._sample_distribution() if not self._should_use_startup_sampling() else np.ones(self.num_tasks) / self.num_tasks
            task_dist = np.nan_to_num(task_dist)  # Convert NaN to 0

            # Reduce amount of per-task basic logging
            log_task_dist = task_dist
            if len(self.tasks) > log_n_tasks and log_n_tasks != -1:
                warnings.warn(
                    f"Too many tasks to log {len(self.tasks)}. Only logging stats for {log_n_tasks} tasks.", stacklevel=2)
                task_dist = task_dist[:log_n_tasks]

            # Add basic logs
            for idx, prob in enumerate(log_task_dist):
                name = self.task_names(self.tasks[idx], idx)
                logs.append((f"curriculum/{name}_prob", prob))

            # Count task with nonzero probability
            nonzero_probs = np.count_nonzero(task_dist)
            logs.append(("curriculum/nonzero_probability_tasks", nonzero_probs))

            # Log entropy
            entropy_task_dist = np.where(task_dist > 0, task_dist, 1e-5)  # Avoid log(0)
            logp = np.log(entropy_task_dist)
            entropy = np.sum(-entropy_task_dist * logp)
            logs.append(("curriculum/entropy", entropy))
            if self.pie_plot_counter % self.pie_plot_interval == 0:
                self.pie_plot_counter = 0
                # Generate task distribution pie chart
                self.plot_pie_chart(task_dist, wandb.run.id, log_n_tasks=log_n_tasks)
                wandb.log({"curriculum/task_distribution": wandb.Image(f"tmp_syllabus_fig_{wandb.run.id}.png"), "global_step": step})

                # Get top 10 tasks
                top_10 = sorted(
                    zip(self.tasks, task_dist),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                top_10_tasks, top_10_probs = zip(*top_10)

                # Log top 10 tasks to wandb table
                try:
                    import wandb
                    self.wandb_table.add_data(step, "\n".join(top_10_tasks),
                                              "\n".join([f"{t:.3f}" for t in top_10_probs]))
                    wandb.log({"curriculum/top_tasks": self.wandb_table, "global_step": step})
                except ImportError:
                    pass
            self.pie_plot_counter += 1

            # Write logs
            for name, prob in logs:
                if use_wandb:
                    writer.log({name: prob})
                else:
                    writer.add_scalar(name, prob, step)
        except Exception as e:
            # No need to crash over logging :)
            warnings.warn(f"Failed to log curriculum stats to wandb. Ignoring error: {e}", stacklevel=2)
        return logs
